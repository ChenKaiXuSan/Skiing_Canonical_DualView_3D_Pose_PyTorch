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

    [Serializable]
    public class SequenceMetaData
    {
        public string subject_id;
        public string action_id;
        public string take_id;
        public string camera_id;
        public int total_frames;
        public int sampled_frames;
        public int joints_count;
        public int pose_every_n_frames;
        public int width;
        public int height;
        public float fov;
        public string created_at_utc;
    }

    [Serializable]
    public class CameraIntrinsicsData
    {
        public int width;
        public int height;
        public float fx;
        public float fy;
        public float cx;
        public float cy;
    }

    [Serializable]
    public class CameraExtrinsicsData
    {
        public float[] t_world_cam_4x4;
        public float[] position_world_xyz;
        public float[] rotation_world_quat_xyzw;
    }

    [Serializable]
    public class DatasetManifestData
    {
        public string updated_at_utc;
        public List<DatasetTakeEntry> takes = new List<DatasetTakeEntry>();
    }

    [Serializable]
    public class DatasetTakeEntry
    {
        public string subject_id;
        public string action_id;
        public string take_id;
        public string take_path;
        public int camera_count;
        public string[] camera_ids;
    }
}

public class OneCameraCaptureFrame : MonoBehaviour
{
    [Header("Output")]
    public string outRootFolder = "SkiDataset";
    public string subjectId = "S001";
    public string actionId = "A001";
    public string takeId = "take_0001";
    public string cameraId = "";

    [Header("Output Naming")]
    [Tooltip("帧目录名前缀，例如 capture_L0_A000")]
    public string captureFolderPrefix = "capture";
    [Tooltip("图片文件名前缀，例如 frame_000001.png")]
    public string imagePrefix = "frame";
    [Tooltip("2D 关键点 npy 文件名前缀，例如 kpt2d.npy")]
    public string kptPrefix = "kpt2d";

    [Tooltip("写出全相机共享的3D关键点（kpt3d/kpt3d.npy），若已存在则默认跳过")]
    public bool exportSharedKpt3d = true;

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

    [Tooltip("当 kpt3d.npy 已存在时是否覆盖")]
    public bool overwriteExistingKpt3d = false;

    [Tooltip("是否在数据集根目录生成/刷新 dataset_manifest.json")]
    public bool exportDatasetManifest = true;

    [Header("NPZ Export")]
    [Tooltip("同时导出 2D 关键点 npz（与 npy 并存）")]
    public bool exportKpt2dNpz = true;

    [Tooltip("同时导出 3D 关键点 npz（与 npy 并存）")]
    public bool exportKpt3dNpz = false;

    // session
    private float baseTime;

    // resources
    private RenderTexture rt;
    private Texture2D tex;

    public bool IsCaptureDone { get; private set; } = false;
    public bool IsCaptureStarted { get; private set; } = false;

    IEnumerator Start()
    {
        IsCaptureStarted = true;
        IsCaptureDone = false;

        if (!autoRunOnPlay) yield break;

        if (cam == null) cam = GetComponent<Camera>();
        if (cam == null)
        {
            Debug.LogError("[OneCameraCaptureFrame] Camera is null.");
            yield break;
        }

        if (string.IsNullOrWhiteSpace(cameraId))
            cameraId = cam.name;

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

        if (targetAnimator == null) targetAnimator = target.GetComponentInChildren<Animator>();
        if (targetAnimator == null)
        {
            Debug.LogError("[OneCameraCaptureFrame] targetAnimator is null.");
            yield break;
        }

        // get total frames
        RuntimeAnimatorController ac = targetAnimator.runtimeAnimatorController;
        if (ac == null)
        {
            Debug.LogError("[OneCameraCaptureFrame] targetAnimator.runtimeAnimatorController is null.");
            yield break;
        }

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

        if (totalFrames <= 0)
        {
            Debug.LogError("[OneCameraCaptureFrame] totalFrames <= 0，无法采样。");
            yield break;
        }

        baseTime = Time.time;
        yield return StartCoroutine(CaptureSequence(totalFrames));

        IsCaptureDone = true;

        // 结束之后退出
    }

    IEnumerator CaptureSequence(int totalFrames)
    {
        string root = Path.Combine(Application.dataPath, "..", outRootFolder);
        string takeRoot = Path.Combine(root, "subjects", subjectId, "actions", actionId, "takes", takeId);
        string metaFolder = Path.Combine(takeRoot, "meta");
        string cameraFolder = Path.Combine(takeRoot, "cameras", cameraId);

        // 使用 captureFolderPrefix 生成帧目录名
        string captureFolderName = string.IsNullOrWhiteSpace(captureFolderPrefix)
            ? cameraId
            : $"{captureFolderPrefix}_{cameraId}";

        string frameRoot = Path.Combine(takeRoot, "frames", captureFolderName);
        string imageFolder = Path.Combine(frameRoot, "images");

        // 使用 kptPrefix 生成 2D 关键点文件名
        string kpt2dFileName = string.IsNullOrWhiteSpace(kptPrefix) ? "kpt2d.npy" : $"{kptPrefix}.npy";
        string kpt2dPath = Path.Combine(frameRoot, kpt2dFileName);

        string kpt3dFolder = Path.Combine(takeRoot, "kpt3d");
        string kpt3dPath = Path.Combine(kpt3dFolder, "kpt3d.npy");

        Directory.CreateDirectory(metaFolder);
        Directory.CreateDirectory(cameraFolder);
        Directory.CreateDirectory(imageFolder);
        Directory.CreateDirectory(kpt3dFolder);

        Debug.Log($"[OneCameraCaptureFrame] Output camera folder: {frameRoot}");

        // allocate RT & texture
        rt = new RenderTexture(captureWidth, captureHeight, 24, RenderTextureFormat.ARGB32);
        rt.Create();
        tex = new Texture2D(captureWidth, captureHeight, TextureFormat.RGBA32, false);

        WriteSequenceMeta(Path.Combine(metaFolder, "sequence.json"), totalFrames, 0);
        WriteCameraIntrinsics(Path.Combine(cameraFolder, "intrinsics.json"));
        WriteCameraExtrinsics(Path.Combine(cameraFolder, "extrinsics.json"));

        // record template
        var rec = new PoseRecord.Data.Frame2DRecordData();
        rec.camera = cam.name;
        rec.joints2d = new List<PoseRecord.Data.Joint2DData>(joints.Length);
        foreach (var j in joints)
            rec.joints2d.Add(new PoseRecord.Data.Joint2DData { name = j.name });

        float oldSpeed = targetAnimator ? targetAnimator.speed : 1f;

        int stride = Mathf.Max(1, poseEveryNFrames);
        int expectedSamples = Mathf.CeilToInt(totalFrames / (float)stride);
        var kpt2dBuffer = new List<float>(expectedSamples * joints.Length * 3);
        var kpt3dBuffer = new List<float>(expectedSamples * joints.Length * 3);
        bool shouldWrite3d = exportSharedKpt3d && (overwriteExistingKpt3d || !File.Exists(kpt3dPath));

        int outputFrameIdx = 0;

        for (int sampleIdx = 0; sampleIdx < totalFrames; sampleIdx += stride)
        {
            if (targetAnimator)
                FreezeAnimatorAtSample(sampleIdx, totalFrames);

            // 传入 imagePrefix
            yield return StartCoroutine(CaptureSingleFrame(sampleIdx, outputFrameIdx, imageFolder, imagePrefix, rec, kpt2dBuffer));

            if (shouldWrite3d)
                AppendKpt3DWorldTJC3(kpt3dBuffer);

            if (targetAnimator)
                targetAnimator.enabled = true;

            outputFrameIdx++;
        }

        WriteKpt2DNpy(kpt2dPath, kpt2dBuffer, outputFrameIdx, joints.Length);
        if (shouldWrite3d)
            WriteKpt3DNpy(kpt3dPath, kpt3dBuffer, outputFrameIdx, joints.Length);

        WriteSequenceMeta(Path.Combine(metaFolder, "sequence.json"), totalFrames, outputFrameIdx);
        if (exportDatasetManifest)
            WriteDatasetManifest(root);


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
        int outputFrameIdx,
        string imageFolder,
        string imgPrefix,
        PoseRecord.Data.Frame2DRecordData rec,
        List<float> kpt2dBuffer
    )
    {
        string safeImagePrefix = string.IsNullOrWhiteSpace(imgPrefix) ? "frame" : imgPrefix;
        Directory.CreateDirectory(imageFolder);
        string imgPath = Path.Combine(imageFolder, $"{safeImagePrefix}_{outputFrameIdx:000000}.png");

        cam.targetTexture = rt;
        yield return new WaitForEndOfFrame();
        cam.Render();

        bool written = false;

        // 先尝试 AsyncGPUReadback
        var req = AsyncGPUReadback.Request(rt, 0, TextureFormat.RGBA32);
        yield return new WaitUntil(() => req.done);

        if (!req.hasError)
        {
            tex.LoadRawTextureData(req.GetData<byte>());
            tex.Apply(false, false);
            File.WriteAllBytes(imgPath, tex.EncodeToPNG());
            written = true;
        }
        else
        {
            Debug.LogWarning($"[OneCameraCaptureFrame] AsyncGPUReadback failed, fallback ReadPixels. frame={outputFrameIdx}");
        }

        // 回退方案：ReadPixels（Mac 上更稳）
        if (!written)
        {
            var prev = RenderTexture.active;
            RenderTexture.active = rt;
            tex.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0, false);
            tex.Apply(false, false);
            RenderTexture.active = prev;

            File.WriteAllBytes(imgPath, tex.EncodeToPNG());
            written = true;
        }

        cam.targetTexture = null;

        if (!File.Exists(imgPath))
            Debug.LogError($"[OneCameraCaptureFrame] PNG write failed: {imgPath}");
        else if (outputFrameIdx == 0)
            Debug.Log($"[OneCameraCaptureFrame] First frame saved: {imgPath}");

        FillRecord(rec, frameIdx);
        AppendKeypointsAsPixelTJC3(kpt2dBuffer);
    }

    void AppendKeypointsAsPixelTJC3(List<float> outBuffer)
    {
        for (int i = 0; i < joints.Length; i++)
        {
            Vector3 vp = cam.WorldToViewportPoint(joints[i].position);
            float conf = (vp.z > 0 && vp.x >= 0 && vp.x <= 1 && vp.y >= 0 && vp.y <= 1) ? 1f : 0f;

            float x01 = flipX ? 1f - vp.x : vp.x;
            float y01 = flipYToTopLeft ? 1f - vp.y : vp.y;

            float x = x01 * (captureWidth - 1);
            float y = y01 * (captureHeight - 1);

            outBuffer.Add(x);
            outBuffer.Add(y);
            outBuffer.Add(conf);
        }
    }

    void WriteSequenceMeta(string path, int totalFrames, int sampledFrames)
    {
        var seq = new PoseRecord.Data.SequenceMetaData
        {
            subject_id = subjectId,
            action_id = actionId,
            take_id = takeId,
            camera_id = cameraId,
            total_frames = totalFrames,
            sampled_frames = sampledFrames,
            joints_count = joints != null ? joints.Length : 0,
            pose_every_n_frames = poseEveryNFrames,
            width = captureWidth,
            height = captureHeight,
            fov = cam != null ? cam.fieldOfView : presetFov,
            created_at_utc = DateTime.UtcNow.ToString("o")
        };

        File.WriteAllText(path, JsonUtility.ToJson(seq, true), Encoding.UTF8);
    }

    void WriteCameraIntrinsics(string path)
    {
        float fovRad = cam.fieldOfView * Mathf.Deg2Rad;
        float fy = 0.5f * captureHeight / Mathf.Tan(0.5f * fovRad);
        float fx = fy * (captureWidth / (float)captureHeight);

        var intr = new PoseRecord.Data.CameraIntrinsicsData
        {
            width = captureWidth,
            height = captureHeight,
            fx = fx,
            fy = fy,
            cx = (captureWidth - 1) * 0.5f,
            cy = (captureHeight - 1) * 0.5f
        };

        File.WriteAllText(path, JsonUtility.ToJson(intr, true), Encoding.UTF8);
    }

    void WriteCameraExtrinsics(string path)
    {
        Matrix4x4 w2c = cam.worldToCameraMatrix;
        Quaternion q = cam.transform.rotation;
        Vector3 p = cam.transform.position;

        var ex = new PoseRecord.Data.CameraExtrinsicsData
        {
            t_world_cam_4x4 = new float[16]
            {
                w2c.m00, w2c.m01, w2c.m02, w2c.m03,
                w2c.m10, w2c.m11, w2c.m12, w2c.m13,
                w2c.m20, w2c.m21, w2c.m22, w2c.m23,
                w2c.m30, w2c.m31, w2c.m32, w2c.m33,
            },
            position_world_xyz = new float[] { p.x, p.y, p.z },
            rotation_world_quat_xyzw = new float[] { q.x, q.y, q.z, q.w }
        };

        File.WriteAllText(path, JsonUtility.ToJson(ex, true), Encoding.UTF8);
    }

    void WriteKpt2DNpy(string path, List<float> data, int t, int j)
    {
        using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write))
        using (var bw = new BinaryWriter(fs))
        {
            bw.Write((byte)0x93);
            bw.Write(Encoding.ASCII.GetBytes("NUMPY"));
            bw.Write((byte)1);
            bw.Write((byte)0);

            string dict = $"{{'descr': '<f4', 'fortran_order': False, 'shape': ({t}, {j}, 3), }}";
            int preambleLen = 10;
            int padLen = 16 - ((preambleLen + dict.Length + 1) % 16);
            if (padLen == 16) padLen = 0;
            string header = dict + new string(' ', padLen) + "\n";

            byte[] headerBytes = Encoding.ASCII.GetBytes(header);
            bw.Write((ushort)headerBytes.Length);
            bw.Write(headerBytes);

            for (int i = 0; i < data.Count; i++)
                bw.Write(data[i]);
        }
    }

    void AppendKpt3DWorldTJC3(List<float> outBuffer)
    {
        for (int i = 0; i < joints.Length; i++)
        {
            Vector3 p = joints[i].position;
            outBuffer.Add(p.x);
            outBuffer.Add(p.y);
            outBuffer.Add(p.z);
        }
    }

    void WriteKpt3DNpy(string path, List<float> data, int t, int j)
    {
        WriteFloatNpy(path, data, t, j, 3);
    }

    void WriteFloatNpy(string path, List<float> data, int d0, int d1, int d2)
    {
        using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write))
        using (var bw = new BinaryWriter(fs))
        {
            bw.Write((byte)0x93);
            bw.Write(Encoding.ASCII.GetBytes("NUMPY"));
            bw.Write((byte)1);
            bw.Write((byte)0);

            string dict = $"{{'descr': '<f4', 'fortran_order': False, 'shape': ({d0}, {d1}, {d2}), }}";
            int preambleLen = 10;
            int padLen = 16 - ((preambleLen + dict.Length + 1) % 16);
            if (padLen == 16) padLen = 0;
            string header = dict + new string(' ', padLen) + "\n";

            byte[] headerBytes = Encoding.ASCII.GetBytes(header);
            bw.Write((ushort)headerBytes.Length);
            bw.Write(headerBytes);

            for (int i = 0; i < data.Count; i++)
                bw.Write(data[i]);
        }
    }

    void WriteDatasetManifest(string datasetRoot)
    {
        string subjectsRoot = Path.Combine(datasetRoot, "subjects");
        var manifest = new PoseRecord.Data.DatasetManifestData
        {
            updated_at_utc = DateTime.UtcNow.ToString("o"),
            takes = CollectTakeEntries(subjectsRoot)
        };

        string manifestPath = Path.Combine(datasetRoot, "dataset_manifest.json");
        File.WriteAllText(manifestPath, JsonUtility.ToJson(manifest, true), Encoding.UTF8);
    }

    List<PoseRecord.Data.DatasetTakeEntry> CollectTakeEntries(string subjectsRoot)
    {
        var list = new List<PoseRecord.Data.DatasetTakeEntry>();
        if (!Directory.Exists(subjectsRoot)) return list;

        foreach (string subjectDir in Directory.GetDirectories(subjectsRoot))
        {
            string subjectIdValue = Path.GetFileName(subjectDir);
            string actionsRoot = Path.Combine(subjectDir, "actions");
            if (!Directory.Exists(actionsRoot)) continue;

            foreach (string actionDir in Directory.GetDirectories(actionsRoot))
            {
                string actionIdValue = Path.GetFileName(actionDir);
                string takesRoot = Path.Combine(actionDir, "takes");
                if (!Directory.Exists(takesRoot)) continue;

                foreach (string takeDir in Directory.GetDirectories(takesRoot))
                {
                    string takeIdValue = Path.GetFileName(takeDir);
                    string camerasRoot = Path.Combine(takeDir, "cameras");
                    string[] cameraDirs = Directory.Exists(camerasRoot)
                        ? Directory.GetDirectories(camerasRoot)
                        : new string[0];

                    string[] cameraIds = new string[cameraDirs.Length];
                    for (int i = 0; i < cameraDirs.Length; i++)
                        cameraIds[i] = Path.GetFileName(cameraDirs[i]);

                    list.Add(new PoseRecord.Data.DatasetTakeEntry
                    {
                        subject_id = subjectIdValue,
                        action_id = actionIdValue,
                        take_id = takeIdValue,
                        take_path = $"subjects/{subjectIdValue}/actions/{actionIdValue}/takes/{takeIdValue}",
                        camera_count = cameraIds.Length,
                        camera_ids = cameraIds
                    });
                }
            }
        }

        return list;
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
