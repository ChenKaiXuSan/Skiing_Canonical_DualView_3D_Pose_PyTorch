using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class CameraRingPlacer : MonoBehaviour
{
    [Header("Character (Auto Assign)")]
    public Transform target;
    public Animator targetAnimator;
    public Transform rootBone;
    public Transform waistCenter;

    [Tooltip("未手动指定 waistCenter 时，优先从 Humanoid 的 Hips 自动获取")]
    public bool autoUseAnimatorHips = true;

    [Tooltip("如果你想手动指定 joints，就填这里（推荐）。否则会自动从 rootBone 扫描")]
    public Transform[] jointsOverride;

    [Tooltip("优先用 SkinnedMeshRenderer.bones（推荐 true）")]
    public bool preferSkinnedMeshBones = true;

    [Header("Ring")]
    [Tooltip("统一相机轨道半径（不区分远近）")]
    public float radius = 3.0f;

    [Tooltip("俯仰角列表（度）：上/中/下三层。负数=俯视到下方，正数=仰视到上方")]
    public float[] pitchAngles = new float[] { -20f, 0f, 20f };

    public int stepDeg = 30;
    public Vector3 lookAtOffset = Vector3.zero;

    [Header("Camera Settings (match inspector)")]
    public float fov = 70f;
    public float nearClip = 0.3f;
    public float farClip = 1000f;
    public int targetDisplay = 1; // Display 2 → index = 1 (不需要多Display就改 0)
    public int cameraDepth = -1;

    [Header("Attach Capture Script")]
    public bool attachCaptureScript = true;

    [Tooltip("不要让每个相机都自动跑！推荐 false，然后用 Manager 串行触发")]
    public bool autoRunOnPlay = true;

    [Tooltip("输出根目录")]
    public string outRootFolder = "SkiDataset";

    [Header("Dataset IDs")]
    public string subjectId = "S001";
    public string actionId = "A001";

    [Tooltip("是否把每个相机输出放到各自子目录 camName/ 下")]
    public bool outputPerCameraSubfolder = true;

    [Header("Cleanup")]
    public string parentName = "CameraRing";
    public bool clearExisting = true;

    [ContextMenu("Create Camera Ring")]
    public void CreateRing()
    {
        if (target == null)
        {
            Debug.LogError("[CameraRingPlacer] Target is null");
            return;
        }

        // 1) Resolve joints ONCE
        Transform[] resolvedJoints = null;
        if (attachCaptureScript)
        {
            resolvedJoints = ResolveJointsOnce();
            if (resolvedJoints == null || resolvedJoints.Length == 0)
            {
                Debug.LogError("[CameraRingPlacer] joints resolved empty. Assign jointsOverride or rootBone.");
                return;
            }
        }

        Transform centerTransform = ResolveCenterTransform();
        Vector3 center = (centerTransform != null ? centerTransform.position : target.position) + lookAtOffset;

        // 2) Parent
        GameObject parent = GameObject.Find(parentName);
        if (parent == null) parent = new GameObject(parentName);

        if (clearExisting)
        {
            for (int i = parent.transform.childCount - 1; i >= 0; --i)
                DestroyImmediate(parent.transform.GetChild(i).gameObject);
        }

        int validStepDeg = Mathf.Max(1, stepDeg);
        int anglesCount = Mathf.CeilToInt(360f / validStepDeg);

        int pitchCount = pitchAngles != null ? pitchAngles.Length : 0;

        if (pitchCount <= 0)
        {
            Debug.LogError("[CameraRingPlacer] pitchAngles is empty. Please set at least one pitch angle.");
            return;
        }

        int createdCount = 0;

        for (int p = 0; p < pitchCount; p++)
        {
            float pitchDeg = pitchAngles[p];
            float pitchRad = pitchDeg * Mathf.Deg2Rad;
            float cosPitch = Mathf.Cos(pitchRad);
            float sinPitch = Mathf.Sin(pitchRad);

            for (int i = 0; i < anglesCount; i++)
            {
                float yawDeg = i * validStepDeg;
                float yawRad = yawDeg * Mathf.Deg2Rad;

                Vector3 dir = new Vector3(
                    Mathf.Sin(yawRad) * cosPitch,
                    sinPitch,
                    Mathf.Cos(yawRad) * cosPitch
                );

                Vector3 pos = center + dir * radius;

                string camId = $"L{p}_A{yawDeg:000}";
                GameObject camGo = new GameObject(camId);
                camGo.transform.SetParent(parent.transform);
                camGo.transform.position = pos;
                camGo.transform.LookAt(center);

                Camera cam = camGo.AddComponent<Camera>();

                // ===== 相机参数：匹配你截图 =====
                cam.clearFlags = CameraClearFlags.SolidColor;
                cam.backgroundColor = Color.black;
                cam.cullingMask = ~0;
                cam.orthographic = false;
                cam.fieldOfView = fov;
                cam.nearClipPlane = nearClip;
                cam.farClipPlane = farClip;
                cam.rect = new Rect(0, 0, 1, 1);
                cam.depth = cameraDepth;
                cam.renderingPath = RenderingPath.UsePlayerSettings;
                cam.useOcclusionCulling = true;
                cam.allowHDR = true;
                cam.allowMSAA = true;
                cam.targetTexture = null;
                cam.targetDisplay = targetDisplay;

                // ✅ 重要：先禁用，避免场景里多相机渲染干扰
                cam.enabled = false;

                // ===== 自动挂载采集脚本（OneCameraCaptureFrame） =====
                if (attachCaptureScript)
                {
                    var cap = camGo.AddComponent<OneCameraCaptureFrame>();

                    // 绑定相机 & 人物
                    cap.cam = cam;
                    cap.target = target;
                    cap.targetAnimator = targetAnimator;

                    // ✅ 关节点：直接塞进去，保证所有相机一致（且不会重复扫描）
                    cap.rootBone = rootBone;
                    cap.autoScanAllChildren = true; // 让它扫描，但我们会覆盖 joints
                    cap.joints = resolvedJoints;

                    // ✅ 不要每台都自动跑
                    cap.autoRunOnPlay = autoRunOnPlay;

                    // 数据集相机ID
                    cap.cameraId = camId;

                    // 数据集根目录（具体相机子目录由 capture 脚本内部组织）
                    cap.outRootFolder = outRootFolder;
                    cap.subjectId = subjectId;
                    cap.actionId = actionId;
                    cap.useClipNameAsActionFolder = true;
                    cap.splitOutputByClip = false;

                    // 前缀
                    cap.captureFolderPrefix = "capture";
                    cap.imagePrefix = "frame";
                    cap.kptPrefix = "kpt2d";

                    // 我们这里已经设置好了 Camera 参数
                    cap.applyInspectorLikePreset = false;
                }

                createdCount++;
            }
        }

        Debug.Log($"[CameraRingPlacer] Created {createdCount} cameras (1 radius x {pitchCount} pitches x {anglesCount} angles, center={(centerTransform != null ? centerTransform.name : target.name)}, enabled=false) + attached capture={attachCaptureScript}");
    }

    Transform ResolveCenterTransform()
    {
        if (waistCenter != null) return waistCenter;

        if (autoUseAnimatorHips)
        {
            Animator animator = targetAnimator;
            if (animator == null && target != null) animator = target.GetComponentInChildren<Animator>();

            if (animator != null && animator.isHuman)
            {
                Transform hips = animator.GetBoneTransform(HumanBodyBones.Hips);
                if (hips != null) return hips;
            }
        }

        return target;
    }

    Transform[] ResolveJointsOnce()
    {
        if (jointsOverride != null && jointsOverride.Length > 0)
        {
            Debug.Log($"[CameraRingPlacer] Using jointsOverride: {jointsOverride.Length}");
            return jointsOverride;
        }

        if (rootBone == null)
        {
            Debug.LogWarning("[CameraRingPlacer] rootBone is null and no jointsOverride provided.");
            return null;
        }

        if (preferSkinnedMeshBones)
        {
            var smr = rootBone.GetComponentInChildren<SkinnedMeshRenderer>();
            if (smr != null && smr.bones != null && smr.bones.Length > 0)
            {
                Debug.Log($"[CameraRingPlacer] Using SkinnedMeshRenderer.bones: {smr.bones.Length}");
                return smr.bones;
            }
        }

        // fallback: all children
        var list = new List<Transform>();
        list.Add(rootBone);
        GetAllChildren(rootBone, list);
        Debug.Log($"[CameraRingPlacer] Using all children under rootBone: {list.Count}");
        return list.ToArray();
    }

    void GetAllChildren(Transform parent, List<Transform> result)
    {
        foreach (Transform child in parent)
        {
            result.Add(child);
            GetAllChildren(child, result);
        }
    }
}
