using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class CameraRingPlacer : MonoBehaviour
{
    [Header("Character (Auto Assign)")]
    public Transform target;
    public Animator targetAnimator;
    public Transform rootBone;

    [Tooltip("如果你想手动指定 joints，就填这里（推荐）。否则会自动从 rootBone 扫描")]
    public Transform[] jointsOverride;

    [Tooltip("优先用 SkinnedMeshRenderer.bones（推荐 true）")]
    public bool preferSkinnedMeshBones = true;

    [Header("Ring")]
    public float radius = 3.0f;
    public float height = 1.0f;
    public int stepDeg = 10;
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
    public string outRootFolder = "RecordingsPose";

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

        Vector3 center = target.position + lookAtOffset;

        // 2) Parent
        GameObject parent = GameObject.Find(parentName);
        if (parent == null) parent = new GameObject(parentName);

        if (clearExisting)
        {
            for (int i = parent.transform.childCount - 1; i >= 0; --i)
                DestroyImmediate(parent.transform.GetChild(i).gameObject);
        }

        int count = Mathf.CeilToInt(360f / stepDeg);

        for (int i = 0; i < count; i++)
        {
            float deg = i * stepDeg;
            float rad = deg * Mathf.Deg2Rad;

            Vector3 pos = center + new Vector3(
                Mathf.Sin(rad) * radius,
                height,
                Mathf.Cos(rad) * radius
            );

            GameObject camGo = new GameObject($"Cam_{deg:000}deg");
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

                // 输出路径按相机分目录
                cap.outRootFolder = outputPerCameraSubfolder
                    ? Path.Combine(outRootFolder, camGo.name)
                    : outRootFolder;

                // 前缀
                cap.captureFolderPrefix = "capture";
                cap.imagePrefix = "frame";
                cap.kptPrefix = "kpt2d";

                // 我们这里已经设置好了 Camera 参数
                cap.applyInspectorLikePreset = false;
            }
        }

        Debug.Log($"[CameraRingPlacer] Created {count} cameras (enabled=false) + attached capture={attachCaptureScript}");
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
