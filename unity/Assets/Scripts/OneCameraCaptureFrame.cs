// OneCameraCaptureFrame.cs
// Single-view capture: one camera, no orbit/rotation.
// - Freeze animator at sampled normalizedTime
// - Render via cam.Render() into RenderTexture
// - Save PNG + 2D keypoints per sampled frame
//
// Put this file under Assets/Scripts/

using System;
using System.IO;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace PoseRecord.Data
{
    [Serializable]
    public class Joint2DData
    {
        public string name;
        public float x, y, conf; // viewport 0~1 (Top-left if flipYToTopLeft=true)
    }

    [Serializable]
    public class Frame2DRecordData
    {
        public int frame_idx;
        public float time;
        public string camera;
        public int width;
        public int height;
        public float angle_deg; // not used (0)
        public string label;
        public List<Joint2DData> joints2d = new List<Joint2DData>();
    }
}

public class OneCameraCaptureFrame : MonoBehaviour
{
    [Header("Auto Run")]
    public bool autoRunOnPlay = true;
    public float startDelaySec = 0.0f;

    [Header("Target")]
    public Transform target;            // 人物 root / pelvis
    public Animator targetAnimator;     // Animator（可选）

    [Tooltip("每隔多少个 Unity 帧采一次姿态")]
    public int poseEveryNFrames = 10;

    [Tooltip("Animator layer index")]
    public int animatorLayer = 0;

    [Header("Joints")]
    [Tooltip("是否自动扫描 rootBone 下的骨骼")]
    public bool autoScanAllChildren = true;

    public Transform rootBone;
    public Transform[] joints;

    [Header("Camera / Capture")]
    public Camera cam;
    public int captureWidth = 1920;
    public int captureHeight = 1080;

    [Header("Keypoint Coordinate")]
    public bool flipYToTopLeft = true;
    public bool flipX = false;

    [Header("Camera Preset")]
    public bool applyInspectorLikePreset = true;
    public float presetFov = 70f;
    public float presetNear = 0.3f;
    public float presetFar = 1000f;
    public int presetTargetDisplay = 0; // 建议 0
    public int presetDepth = -1;

    [Header("Output")]
    public string outRootFolder = "RecordingsPose";
    public string captureFolderPrefix = "capture";
    public string imagePrefix = "frame";
    public string kptPrefix = "kpt2d";

    // session
    private float baseTime;

    // resources
    private RenderTexture rt;
    private Texture2D tex;

    IEnumerator Start()
    {
        if (!autoRunOnPlay) yield break;

        if (cam == null) cam = GetComponent<Camera>();
        if (cam == null)
        {
            Debug.LogError("[OneCameraCaptureFrame] Camera is null.");
            yield break;
        }

        if (target == null)
        {
            Debug.LogError("[OneCameraCaptureFrame] target is null.");
            yield break;
        }

        if (applyInspectorLikePreset)
            ApplyCameraPresetLikeInspector();

        // resolve joints
        if (autoScanAllChildren && rootBone != null)
        {
            var list = new List<Transform>();
            var smr = rootBone.GetComponentInChildren<SkinnedMeshRenderer>();
            if (smr != null && smr.bones != null && smr.bones.Length > 0)
                list.AddRange(smr.bones);
            else
            {
                list.Add(rootBone);
                GetAllChildren(rootBone, list);
            }
            joints = list.ToArray();
        }

        if (joints == null || joints.Length == 0)
        {
            Debug.LogError("[OneCameraCaptureFrame] joints is empty.");
            yield break;
        }

        if (startDelaySec > 0f)
            yield return new WaitForSeconds(startDelaySec);

        if (targetAnimator == null) targetAnimator = GetComponent<Animator>();

        // get total frames
        RuntimeAnimatorController ac = targetAnimator.runtimeAnimatorController;

        int totalFrames = 0;

        // 遍历 Animator 中引用的所有动画片段
        foreach (AnimationClip clip in ac.animationClips)
        {
            // clip.length 是秒数
            // clip.frameRate 是该动画制作时的采样率（通常是 30 或 60）
            // 使用 Mathf.RoundToInt 确保浮点数转换精确
            int frames = Mathf.RoundToInt(clip.length * clip.frameRate);
            totalFrames += frames;

            Debug.Log($"动画: {clip.name} | 时长: {clip.length}s | 帧率: {clip.frameRate} | 帧数: {frames}");
        }

        Debug.Log($"<color=yellow>所有动画的总帧数: {totalFrames}</color>");

        baseTime = Time.time;
        yield return StartCoroutine(CaptureSequence(totalFrames));

        // 结束之后退出
    }

    IEnumerator CaptureSequence(int totalFrames)
    {
        // output folder
        string root = Path.Combine(Application.dataPath, "..", outRootFolder);
        Directory.CreateDirectory(root);


        string capFolder = Path.Combine(
            root, $"{captureFolderPrefix}"
        );
        Directory.CreateDirectory(capFolder);

        Debug.Log($"[OneCameraCaptureFrame] Output: {capFolder}");

        // allocate RT & texture
        rt = new RenderTexture(captureWidth, captureHeight, 24, RenderTextureFormat.ARGB32);
        rt.Create();
        tex = new Texture2D(captureWidth, captureHeight, TextureFormat.RGBA32, false);

        // record template
        var rec = new PoseRecord.Data.Frame2DRecordData();
        rec.camera = cam.name;
        rec.joints2d = new List<PoseRecord.Data.Joint2DData>(joints.Length);
        foreach (var j in joints)
            rec.joints2d.Add(new PoseRecord.Data.Joint2DData { name = j.name });

        float oldSpeed = targetAnimator ? targetAnimator.speed : 1f;

        int sampleIdx = 0;

        while (sampleIdx < totalFrames)
        {
            // 1) 跳过不采样的帧
            if (sampleIdx % poseEveryNFrames != 0)
            {
                sampleIdx++;
                continue;
            }

            // 2) 冻结到第 sampleIdx 个采样姿态
            if (targetAnimator)
                FreezeAnimatorAtSample(sampleIdx, totalFrames);

            // 3) 输出目录
            string frameFolder = Path.Combine(capFolder, $"frame_{sampleIdx:000000}");
            Directory.CreateDirectory(frameFolder);

            // 4) 单视角采集
            yield return StartCoroutine(CaptureSingleFrame(sampleIdx, frameFolder, rec));

            // 5) 解冻 animator（给下一轮用）
            if (targetAnimator)
                targetAnimator.enabled = true;

            sampleIdx++;
        }


        if (targetAnimator) targetAnimator.speed = oldSpeed;

        rt.Release();
        Destroy(rt);
        Destroy(tex);

        Debug.Log("[OneCameraCaptureFrame] Done.");
    }

    void FreezeAnimatorAtSample(int sampleIdx, int totalFrames)
    {
        // 1. 启用 Animator 并将播放速度设为 0。
        // 这样做是为了接管控制权，确保动画不会因为时间流逝而自动播放。
        targetAnimator.enabled = true;
        targetAnimator.speed = 0f;

        // 2. 立即手动更新一次 Animator（增量时间为 0）。
        // 这一步是为了确保 Animator 当前的状态机数据是最新的。
        targetAnimator.Update(0f);

        // 3. 获取指定动画层（animatorLayer）当前正在播放的状态信息。
        var st = targetAnimator.GetCurrentAnimatorStateInfo(animatorLayer);


        // 4. 获取该动画状态的唯一标识符（全路径哈希值）。
        // 后面我们需要通过这个 hash 值告诉 Play 函数我们要跳回到哪个动画状态。
        int hash = st.fullPathHash;

        // 5. 容错处理：如果获取到的 hash 为 0（通常表示 Animator 还没初始化完成或没开始播放）。
        if (hash == 0)
        {
            // 强制播放该层的第 0 个默认状态，时间点设为 0（起始点）。
            targetAnimator.Play(0, animatorLayer, 0f);
            targetAnimator.Update(0f);

            // 重新获取正确的状态 hash 值。
            hash = targetAnimator.GetCurrentAnimatorStateInfo(animatorLayer).fullPathHash;
        }

        float totalFramesF = (float)totalFrames;
        // 6. 设置归一化时间（Normalized Time）。
        // 这是这段代码最关键的地方：Play 函数的第三个参数要求是 0.0 到 1.0 之间的浮点数。
        // 0 代表动画开头，1 代表动画结束。
        // 假设总共要采样 100 个姿态
        // 3. 计算归一化时间 t01 (0.0 到 1.0)
        // 使用 Mathf.Clamp 确保值不会超过 1.0，防止由于浮点数误差导致的越界
        float t01 = Mathf.Clamp01(sampleIdx / totalFramesF);

        // 7. 核心操作：调用 Play 函数。
        // 作用：让 Animator 立即跳转到指定状态（hash）的指定时间点（t01）。
        targetAnimator.Play(hash, animatorLayer, t01);

        // 8. 再次手动调用 Update(0)。
        // 作用：让刚才 Play 函数设置的“跳转”立即生效，更新骨骼节点的位置。
        targetAnimator.Update(0f);

        // 9. 最后禁用 Animator。
        // 这样角色就会彻底固定在当前的姿态，直到下一次调用此函数。
        targetAnimator.enabled = false;

        Debug.Log($"跳转到帧: {sampleIdx}, 对应时间点: {t01:P2}");
    }

    IEnumerator CaptureSingleFrame(
        int frameIdx,
        string frameFolder,
        PoseRecord.Data.Frame2DRecordData rec
    )
    {
        string imgPath = Path.Combine(
            frameFolder, $"{imagePrefix}_frame_{frameIdx:000000}.png"
        );
        string jsonPath = Path.Combine(
            frameFolder, $"{kptPrefix}_frame_{frameIdx:000000}.json"
        );

        cam.targetTexture = rt;
        yield return new WaitForEndOfFrame();
        cam.Render();

        var req = AsyncGPUReadback.Request(rt, 0, TextureFormat.RGBA32);
        yield return new WaitUntil(() => req.done);

        // 保存图片
        if (!req.hasError)
        {
            tex.LoadRawTextureData(req.GetData<byte>());
            tex.Apply(false, false);
            // FlipTextureVertical(tex); // ！ 不需要翻转，已经是正确方向了
            File.WriteAllBytes(imgPath, tex.EncodeToPNG());
        }

        cam.targetTexture = null;

        // 保存 keypoints
        FillRecord(rec, frameIdx);
        File.WriteAllText(jsonPath, JsonUtility.ToJson(rec, false), Encoding.UTF8);
    }

    void FillRecord(PoseRecord.Data.Frame2DRecordData rec, int frameIdx)
    {
        rec.frame_idx = frameIdx;
        rec.time = Time.time - baseTime;
        rec.width = captureWidth;
        rec.height = captureHeight;
        rec.angle_deg = 0f;
        rec.label = GetClipLabel();

        for (int i = 0; i < joints.Length; i++)
        {
            Vector3 vp = cam.WorldToViewportPoint(joints[i].position);
            var jd = rec.joints2d[i];

            jd.conf = (vp.z > 0 && vp.x >= 0 && vp.x <= 1 && vp.y >= 0 && vp.y <= 1) ? 1f : 0f;

            float x = flipX ? 1f - vp.x : vp.x;
            float y = flipYToTopLeft ? 1f - vp.y : vp.y;

            jd.x = x;
            jd.y = y;
        }
    }

    string GetClipLabel()
    {
        if (!targetAnimator) return "NoAnimator";
        var clips = targetAnimator.GetCurrentAnimatorClipInfo(animatorLayer);
        if (clips == null || clips.Length == 0) return "NoClip";
        return clips[0].clip.name;
    }

    void ApplyCameraPresetLikeInspector()
    {
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = Color.black;
        cam.fieldOfView = presetFov;
        cam.nearClipPlane = presetNear;
        cam.farClipPlane = presetFar;
        cam.targetDisplay = Mathf.Max(0, presetTargetDisplay);
        cam.depth = presetDepth;
    }

    static void FlipTextureVertical(Texture2D tex)
    {
        int w = tex.width, h = tex.height;
        Color[] p = tex.GetPixels();
        for (int y = 0; y < h / 2; y++)
        {
            int y2 = h - 1 - y;
            for (int x = 0; x < w; x++)
            {
                int i1 = y * w + x;
                int i2 = y2 * w + x;
                (p[i1], p[i2]) = (p[i2], p[i1]);
            }
        }
        tex.SetPixels(p);
        tex.Apply(false, false);
    }

    void GetAllChildren(Transform t, List<Transform> list)
    {
        foreach (Transform c in t)
        {
            list.Add(c);
            GetAllChildren(c, list);
        }
    }
}
