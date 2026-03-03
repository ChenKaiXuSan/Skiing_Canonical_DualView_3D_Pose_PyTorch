// using System;
// using System.IO;
// using System.Text;
// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine;
// using UnityEngine.Rendering;

// namespace PoseRecord.Data
// {
//     [Serializable]
//     public class Joint2DData
//     {
//         public string name;
//         public float x, y, conf; // viewport 0~1 (Top-left)
//     }

//     [Serializable]
//     public class Frame2DRecordData
//     {
//         public int capture_idx;
//         public int frame_idx;
//         public float time;
//         public string camera;
//         public int width;
//         public int height;
//         public float angle_deg;
//         public string label;
//         public List<Joint2DData> joints2d = new List<Joint2DData>();
//     }
// }

// public class AutoOrbitCaptureKpt : MonoBehaviour
// {
//     [Header("Pose Step (per dataset frame)")]
//     [Tooltip("true: 每一帧手动推进 Animator，然后冻结拍一圈（推荐，最稳定）")]
//     public bool manualStepAnimator = true;

//     [Tooltip("每个 frame 推进多少秒，比如 1/30 或 1/60")]
//     public float poseStepSec = 1f / 60f;

//     [Tooltip("Animator layer index for label + stepping")]
//     public int animatorLayer = 0;

//     [Header("Orbit Pivot Mode")]
//     [Tooltip("true: pivot 跟随 target 的 XZ（绕人转）; false: pivot 固定在初始位置（世界固定轨道）")]
//     public bool pivotFollowsTargetXZ = false;

//     [Header("Auto Run")]
//     public bool autoRunOnPlay = true;
//     public float startDelaySec = 0.2f;

//     [Header("Frame Loop")]
//     [Tooltip("保存するフレーム数。-1 なら無限。まず 10 くらいでテスト推奨。")]
//     public int maxFrames = 10;
//     [Tooltip("各フレーム開始時に待つフレーム数（アニメ・物理の安定用）")]
//     public int warmupFramesPerFrame = 0;

//     [Header("Target / Orbit")]
//     public Transform target;
//     public Vector3 lookAtOffset = Vector3.zero;
//     public Vector3 orbitAxis = Vector3.up;
//     public float stepAngleDeg = 10f;

//     [Header("Freeze Pose (Optional)")]
//     public Animator targetAnimator;
//     public bool freezeAnimatorDuringCapture = false;

//     [Header("Joints")]
//     public bool autoScanAllChildren = true;
//     public Transform rootBone;
//     public Transform[] joints;

//     [Header("Camera / Capture Size")]
//     public Camera cam;
//     public int captureWidth = 1920;
//     public int captureHeight = 1080;

//     [Header("Keypoint Coordinate")]
//     [Tooltip("viewport y をTop-leftにする（CV風）。推奨 true")]
//     public bool flipYToTopLeft = true;
//     [Tooltip("必要な時だけ左右反転")]
//     public bool flipX = false;

//     [Header("Output")]
//     public string outRootFolder = "RecordingsPose";
//     public string captureFolderPrefix = "capture";
//     public string imagePrefix = "frame";
//     public string kptPrefix = "kpt2d";
//     public bool appendTimestampToCaptureFolder = true;

//     private int captureIndex = 0;
//     private float baseTime;

//     // resources reused in a capture session
//     private RenderTexture rt;
//     private Texture2D tex;

//     // ===== Locked orbit params (constant across frames) =====
//     private float lockedPivotY;
//     private float lockedCamY;
//     private Vector3 lockedOffsetXZ;   // horizontal radius vector (Y=0)
//     private float lockedPitch0;
//     private float lockedRoll0;
//     private Vector3 orbitAxisN;

//     private Vector3 lockedPivot0;  // 世界固定 pivot

//     [Header("Animator Stepping (RECOMMENDED)")]
//     [Tooltip("true: 用 normalizedTime 精确推进动画（最稳定）")]
//     public bool stepByNormalizedTime = true;

//     IEnumerator Start()
//     {
//         if (!autoRunOnPlay) yield break;

//         if (cam == null) cam = GetComponent<Camera>();
//         if (cam == null)
//         {
//             Debug.LogError("[AutoOrbitCaptureKpt] Camera is null. Attach script to a Camera or assign cam.");
//             yield break;
//         }

//         if (target == null)
//         {
//             Debug.LogError("[AutoOrbitCaptureKpt] target is null. Assign target (character pelvis/root).");
//             yield break;
//         }

//         // auto-scan joints
//         if (autoScanAllChildren && rootBone != null)
//         {
//             var list = new List<Transform>();
//             var smr = rootBone.GetComponentInChildren<SkinnedMeshRenderer>();
//             if (smr != null && smr.bones != null && smr.bones.Length > 0)
//                 list.AddRange(smr.bones);
//             else
//             {
//                 list.Add(rootBone);
//                 GetAllChildren(rootBone, list);
//             }
//             joints = list.ToArray();
//         }

//         if (joints == null || joints.Length == 0)
//         {
//             Debug.LogError("[AutoOrbitCaptureKpt] joints is empty. Assign rootBone or joints manually.");
//             yield break;
//         }

//         if (startDelaySec > 0f) yield return new WaitForSeconds(startDelaySec);

//         baseTime = Time.time;

//         yield return StartCoroutine(CaptureSequence());
//     }

//     IEnumerator CaptureSequence()
//     {
//         // ---------------------------
//         // 1) Prepare output folder
//         // ---------------------------
//         string root = Path.Combine(Application.dataPath, "..", outRootFolder);
//         Directory.CreateDirectory(root);

//         string timeStr = appendTimestampToCaptureFolder ? "_" + DateTime.Now.ToString("yyyyMMdd_HHmmss") : "";
//         string capFolder = Path.Combine(root, $"{captureFolderPrefix}_{captureIndex:000000}{timeStr}");
//         Directory.CreateDirectory(capFolder);
//         Debug.Log($"[AutoOrbitCaptureKpt] Output folder: {capFolder}");

//         // ---------------------------
//         // 2) Allocate RT & Texture (once)
//         // ---------------------------
//         rt = new RenderTexture(captureWidth, captureHeight, 24, RenderTextureFormat.ARGB32);
//         rt.Create();
//         tex = new Texture2D(captureWidth, captureHeight, TextureFormat.RGBA32, false);

//         // ---------------------------
//         // 3) Save initial camera pose
//         // ---------------------------
//         Vector3 initPos = cam.transform.position;
//         Quaternion initRot = cam.transform.rotation;

//         // ---------------------------
//         // 4) Lock orbit parameters ONCE
//         //    !!! 注意：lockedPivot0 必须在 lockedPivotY 赋值之后计算
//         // ---------------------------
//         orbitAxisN = orbitAxis.sqrMagnitude < 1e-6f ? Vector3.up : orbitAxis.normalized;

//         Vector3 e0 = initRot.eulerAngles;
//         lockedPitch0 = e0.x;
//         lockedRoll0 = e0.z;

//         lockedPivotY = target.position.y + lookAtOffset.y;  // pivot height fixed
//         lockedCamY = cam.transform.position.y;              // camera height fixed

//         lockedPivot0 = new Vector3(
//             target.position.x + lookAtOffset.x,
//             lockedPivotY,
//             target.position.z + lookAtOffset.z
//         );

//         Vector3 cam0 = new Vector3(cam.transform.position.x, lockedCamY, cam.transform.position.z);
//         lockedOffsetXZ = cam0 - lockedPivot0;
//         lockedOffsetXZ.y = 0f;

//         if (lockedOffsetXZ.sqrMagnitude < 1e-8f)
//             lockedOffsetXZ = new Vector3(0f, 0f, -3f);

//         Debug.Log($"[AutoOrbitCaptureKpt] Locked orbit: pivot={lockedPivot0}, camY={lockedCamY:F3}, radius={lockedOffsetXZ.magnitude:F3}");

//         // ---------------------------
//         // 5) Record template (reuse)
//         // ---------------------------
//         var rec = new PoseRecord.Data.Frame2DRecordData();
//         rec.camera = cam.name;
//         rec.joints2d = new List<PoseRecord.Data.Joint2DData>(joints.Length);
//         foreach (var j in joints)
//             rec.joints2d.Add(new PoseRecord.Data.Joint2DData { name = j != null ? j.name : "null" });

//         // ---------------------------
//         // 6) Animator speed cache
//         // ---------------------------
//         float oldSpeed = 1f;
//         if (targetAnimator != null) oldSpeed = targetAnimator.speed;

//         // ---------------------------
//         // 7) Frame loop: 先推进姿态 -> 冻结 -> 拍一圈 -> 解冻/准备下一帧
//         // ---------------------------
//         int frameIdx = 0;
//         while (maxFrames < 0 || frameIdx < maxFrames)
//         {
//             // warmup (optional)
//             for (int k = 0; k < warmupFramesPerFrame; k++)
//                 yield return new WaitForEndOfFrame();

//             if (frameIdx % 60 != 0)
//             {
//                 frameIdx++;
//                 continue;
//             }

//             // (A) advance pose to this dataset frame  ✅必须在拍摄前做
//             if (targetAnimator != null && manualStepAnimator)
//             {
//                 targetAnimator.enabled = true;     // ✅确保可评估
//                 targetAnimator.speed = 0f;
//                 targetAnimator.Update(0f);         // ✅清一次内部状态（很重要，尤其有过 enabled=false）

//                 float t01 = frameIdx / 10;

//                 var st = targetAnimator.GetCurrentAnimatorStateInfo(animatorLayer);
//                 int stateHash = st.fullPathHash;
//                 if (stateHash == 0)
//                 {
//                     targetAnimator.Play(0, animatorLayer, 0f);
//                     targetAnimator.Update(0f);
//                     st = targetAnimator.GetCurrentAnimatorStateInfo(animatorLayer);
//                     stateHash = st.fullPathHash;
//                 }

//                 // 这里调整动画进度
//                 targetAnimator.Play(stateHash, animatorLayer, t01);
//                 targetAnimator.Update(0f);         // ✅立刻评估骨骼

//                 targetAnimator.enabled = false;    // ✅冻结：绕圈期间绝不会动
//             }


//             // (B) capture orbit for this frame (pose is frozen now)
//             string frameFolder = Path.Combine(capFolder, $"frame_{frameIdx:000000}");
//             Directory.CreateDirectory(frameFolder);
//             yield return StartCoroutine(CaptureOrbitForCurrentFrame(frameIdx, frameFolder, rec));
//             if (targetAnimator != null)
//                 targetAnimator.enabled = true;

//             frameIdx++;
//         }

//         // ---------------------------
//         // 8) Restore camera
//         // ---------------------------
//         cam.transform.position = initPos;
//         cam.transform.rotation = initRot;

//         // restore animator speed
//         if (targetAnimator != null) targetAnimator.speed = oldSpeed;

//         // ---------------------------
//         // 9) Cleanup
//         // ---------------------------
//         if (rt != null) { rt.Release(); Destroy(rt); rt = null; }
//         if (tex != null) { Destroy(tex); tex = null; }

//         Debug.Log($"[AutoOrbitCaptureKpt] Session Saved: {capFolder}");
//         captureIndex++;
//     }


//     IEnumerator CaptureOrbitForCurrentFrame(int frameIdx, string frameFolder, PoseRecord.Data.Frame2DRecordData rec)
//     {
//         // ✅ pivot mode
//         Vector3 pivot;
//         if (pivotFollowsTargetXZ)
//         {
//             pivot = new Vector3(
//                 target.position.x + lookAtOffset.x,
//                 lockedPivotY,
//                 target.position.z + lookAtOffset.z
//             );
//         }
//         else
//         {
//             pivot = lockedPivot0; // world-fixed
//         }

//         float steps = 360 / (int)stepAngleDeg;

//         for (int s = 0; s < steps; s++)
//         {
//             float ang = (s * stepAngleDeg) % 360f;

//             Quaternion q = Quaternion.AngleAxis(ang, orbitAxisN);
//             Vector3 rotatedOffset = q * lockedOffsetXZ;

//             Vector3 camPos = pivot + rotatedOffset;
//             camPos.y = lockedCamY;
//             cam.transform.position = camPos;

//             Vector3 dir = pivot - cam.transform.position;
//             dir.y = 0f;
//             if (dir.sqrMagnitude > 1e-8f)
//             {
//                 float yaw = Quaternion.LookRotation(dir.normalized, Vector3.up).eulerAngles.y;
//                 cam.transform.rotation = Quaternion.Euler(lockedPitch0, yaw, lockedRoll0);
//             }

//             // wait render
//             yield return new WaitForEndOfFrame();

//             // angle folder
//             string angleFolder = Path.Combine(frameFolder, $"angle_{(int)ang:000}");
//             Directory.CreateDirectory(angleFolder);

//             string imgName = $"{imagePrefix}_frame_{frameIdx:000000}_angle_{(int)ang:000}.png";
//             string jsonName = $"{kptPrefix}_frame_{frameIdx:000000}_angle_{(int)ang:000}.json";

//             string imgPath = Path.Combine(angleFolder, imgName);
//             string jsonPath = Path.Combine(angleFolder, jsonName);

//             // --- capture to RT ---
//             ScreenCapture.CaptureScreenshotIntoRenderTexture(rt);

//             // --- readback ---
//             var req = AsyncGPUReadback.Request(rt, 0, TextureFormat.RGBA32);
//             yield return new WaitUntil(() => req.done);

//             if (req.hasError)
//             {
//                 Debug.LogError($"[AutoOrbitCaptureKpt] AsyncGPUReadback error frame={frameIdx} angle={ang}");
//             }
//             else
//             {
//                 var data = req.GetData<byte>();
//                 tex.LoadRawTextureData(data);
//                 tex.Apply(false, false);

//                 // flip image to Top-left (CV style)
//                 FlipTextureVertical(tex);

//                 byte[] png = tex.EncodeToPNG();
//                 if (png == null || png.Length < 100)
//                 {
//                     Debug.LogError($"[AutoOrbitCaptureKpt] EncodeToPNG failed/too small frame={frameIdx} angle={ang}");
//                 }
//                 else
//                 {
//                     File.WriteAllBytes(imgPath, png);
//                 }
//             }

//             // --- json ---
//             FillRecord(rec, frameIdx, ang, captureWidth, captureHeight, pivot);
//             File.WriteAllText(jsonPath, JsonUtility.ToJson(rec, false), Encoding.UTF8);
//         }
//     }

//     void FillRecord(PoseRecord.Data.Frame2DRecordData rec, int frameIdx, float angDeg, int width, int height, Vector3 pivot)
//     {
//         rec.capture_idx = captureIndex;
//         rec.frame_idx = frameIdx;
//         rec.time = Time.time - baseTime;
//         rec.width = width;
//         rec.height = height;
//         rec.angle_deg = angDeg;
//         rec.label = GetClipLabel();

//         if (rec.joints2d == null) rec.joints2d = new List<PoseRecord.Data.Joint2DData>();
//         if (rec.joints2d.Count != joints.Length)
//         {
//             rec.joints2d.Clear();
//             foreach (var j in joints)
//                 rec.joints2d.Add(new PoseRecord.Data.Joint2DData { name = j != null ? j.name : "null" });
//         }

//         for (int i = 0; i < joints.Length; i++)
//         {
//             if (joints[i] == null) continue;

//             Vector3 vp = cam.WorldToViewportPoint(joints[i].position);
//             var jd = rec.joints2d[i];

//             bool inFront = vp.z > 0f;
//             bool inView = (vp.x >= 0f && vp.x <= 1f && vp.y >= 0f && vp.y <= 1f);
//             jd.conf = (inFront && inView) ? 1f : 0f;

//             float x = vp.x;
//             float y = vp.y;

//             if (flipX) x = 1f - x;
//             if (flipYToTopLeft) y = 1f - y;

//             jd.x = x;
//             jd.y = y;
//         }
//     }

//     string GetClipLabel()
//     {
//         if (targetAnimator == null) return "NoAnimator";
//         // var clips = targetAnimator.GetCurrentAnimatorClipInfo(0);

//         var clips = targetAnimator.GetCurrentAnimatorClipInfo(animatorLayer);

//         if (clips == null || clips.Length == 0) return "NoClip";

//         int best = 0;
//         float w = clips[0].weight;
//         for (int i = 1; i < clips.Length; i++)
//         {
//             if (clips[i].weight > w) { w = clips[i].weight; best = i; }
//         }
//         return clips[best].clip != null ? clips[best].clip.name : "NoClip";
//     }

//     static void FlipTextureVertical(Texture2D tex)
//     {
//         int w = tex.width;
//         int h = tex.height;
//         Color[] pixels = tex.GetPixels();

//         for (int y = 0; y < h / 2; y++)
//         {
//             int y2 = h - 1 - y;
//             for (int x = 0; x < w; x++)
//             {
//                 int i1 = y * w + x;
//                 int i2 = y2 * w + x;
//                 (pixels[i1], pixels[i2]) = (pixels[i2], pixels[i1]);
//             }
//         }
//         tex.SetPixels(pixels);
//         tex.Apply(false, false);
//     }

//     void GetAllChildren(Transform parent, List<Transform> result)
//     {
//         foreach (Transform child in parent)
//         {
//             result.Add(child);
//             GetAllChildren(child, result);
//         }
//     }
// }
