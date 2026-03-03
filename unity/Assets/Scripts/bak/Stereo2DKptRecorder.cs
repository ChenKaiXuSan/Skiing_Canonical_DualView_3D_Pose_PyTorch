// using System;
// using System.IO;
// using System.Text;
// using System.Collections.Generic;
// using UnityEngine;
// using PoseRecord.Data.Frame2DRecordData;

// var Frame2DRecord = PoseRecord.Data.Frame2DRecordData;;
// var Joint2D = PoseRecord.Data.Joint2DData;
// public class Stereo2DKptRecorder : MonoBehaviour
// {
//     [Header("Auto Scan Settings")]
//     public bool autoScanAllChildren = true;
//     public Transform rootBone;

//     [Header("Stereo Cameras")]
//     public Camera camLeft;
//     public Camera camRight;

//     [Header("Manual Setup (if auto scan is off)")]
//     public Transform[] joints;

//     [Header("Output Settings")]
//     public string outRootFolder = "RecordingsPose";
//     [Tooltip("ファイル名の基本名（カメラ名が自動で付与されます）")]
//     public string outputFileName = "kpt2d";
//     [Tooltip("ファイル名に日付と時刻を含めるか")]
//     public bool appendTimestamp = true;

//     [Header("Coordinate Settings")]
//     [Tooltip("WorldToScreenPoint の (0,0)=BottomLeft を (0,0)=TopLeft にする")]
//     public bool flipYToTopLeft = true;

//     [Tooltip("Recorder/Video Capture 側で上下反転する場合の補正（必要な時だけON）")]
//     public bool recorderFlipVerticalOn = false;

//     [Header("Flush Settings")]
//     public int flushEveryNFrames = 60;

//     [Header("Animator Label Settings")]
//     [Tooltip("动作来源 Animator（必填）")]
//     public Animator targetAnimator;

//     [Tooltip("Animator layer index")]
//     public int animatorLayer = 0;

//     [Tooltip("如果在过渡(transition)中，优先使用 Next state 的 clip 名称")]
//     public bool preferNextClipWhenTransition = true;

//     private StreamWriter wLeft;
//     private StreamWriter wRight;
//     private int frameIdx = 0;
//     private float baseTime;

//     private readonly Frame2DRecord recLeft = new Frame2DRecord();
//     private readonly Frame2DRecord recRight = new Frame2DRecord();

//     void Start()
//     {
//         Application.targetFrameRate = 60;

//         // --- Auto scan joints ---
//         if (autoScanAllChildren && rootBone != null)
//         {
//             var list = new List<Transform>();
//             var smr = rootBone.GetComponentInChildren<SkinnedMeshRenderer>();
//             if (smr != null && smr.bones != null && smr.bones.Length > 0)
//             {
//                 list.AddRange(smr.bones);
//             }
//             else
//             {
//                 list.Add(rootBone);
//                 GetAllChildren(rootBone, list);
//             }
//             joints = list.ToArray();
//         }

//         if (camLeft == null || camRight == null || joints == null || joints.Length == 0 || targetAnimator == null)
//         {
//             Debug.LogError("[Stereo2D] camLeft / camRight / joints / targetAnimator 未正确设置。");
//             enabled = false;
//             return;
//         }

//         // --- Output dir & file name ---
//         string timeStr = appendTimestamp ? "_" + DateTime.Now.ToString("yyyyMMdd_HHmmss") : "";
//         string outDir = Path.Combine(Application.dataPath, "..", outRootFolder);
//         Directory.CreateDirectory(outDir);

//         wLeft = CreateWriter(outDir, camLeft.name, timeStr);
//         wRight = CreateWriter(outDir, camRight.name, timeStr);

//         InitializeRecord(recLeft, camLeft.name);
//         InitializeRecord(recRight, camRight.name);

//         baseTime = Time.time;
//     }

//     StreamWriter CreateWriter(string root, string camName, string timeStr)
//     {
//         string d = Path.Combine(root, $"cam_{camName}");
//         Directory.CreateDirectory(d);
//         // Example: kpt2d_LeftCam_20260202_101500.jsonl
//         string fileName = $"{outputFileName}_{camName}.jsonl";
//         return new StreamWriter(Path.Combine(d, fileName), false, Encoding.UTF8);
//     }

//     void InitializeRecord(Frame2DRecord rec, string camName)
//     {
//         rec.camera = camName;
//         rec.joints2d.Clear();
//         foreach (var j in joints) rec.joints2d.Add(new Joint2D { name = j != null ? j.name : "null" });
//     }

//     void LateUpdate()
//     {
//         if (wLeft == null || wRight == null) return;

//         float t = Time.time - baseTime;

//         // ✅ per-frame clip label
//         string label = GetClipLabel();

//         UpdateAndWrite(camLeft, wLeft, recLeft, t, label);
//         UpdateAndWrite(camRight, wRight, recRight, t, label);

//         frameIdx++;

//         if (flushEveryNFrames > 0 && (frameIdx % flushEveryNFrames) == 0)
//         {
//             wLeft.Flush();
//             wRight.Flush();
//         }
//     }

//     /// <summary>
//     /// Get per-frame action label from AnimationClip.name.
//     /// - For BlendTree / transitions, selects the clip with the highest weight.
//     /// </summary>
//     string GetClipLabel()
//     {
//         if (targetAnimator == null) return "AnimatorNull";

//         // Transition: optionally prefer next clip
//         if (preferNextClipWhenTransition && targetAnimator.IsInTransition(animatorLayer))
//         {
//             var nextClips = targetAnimator.GetNextAnimatorClipInfo(animatorLayer);
//             string next = GetTopWeightedClipName(nextClips);
//             if (!string.IsNullOrEmpty(next)) return next;
//         }

//         var curClips = targetAnimator.GetCurrentAnimatorClipInfo(animatorLayer);
//         string cur = GetTopWeightedClipName(curClips);
//         if (!string.IsNullOrEmpty(cur)) return cur;

//         // Fallback: if transition but preferNextClipWhenTransition=false, try next
//         if (targetAnimator.IsInTransition(animatorLayer))
//         {
//             var nextClips = targetAnimator.GetNextAnimatorClipInfo(animatorLayer);
//             string next = GetTopWeightedClipName(nextClips);
//             if (!string.IsNullOrEmpty(next)) return next;
//         }

//         return "NoClip";
//     }

//     string GetTopWeightedClipName(AnimatorClipInfo[] clips)
//     {
//         if (clips == null || clips.Length == 0) return null;

//         int bestIdx = 0;
//         float bestW = clips[0].weight;
//         for (int i = 1; i < clips.Length; i++)
//         {
//             if (clips[i].weight > bestW)
//             {
//                 bestW = clips[i].weight;
//                 bestIdx = i;
//             }
//         }

//         var clip = clips[bestIdx].clip;
//         return clip != null ? clip.name : null;
//     }

//     void UpdateAndWrite(Camera cam, StreamWriter w, Frame2DRecord rec, float t, string label)
//     {
//         int width, height;

//         if (cam.targetTexture != null)
//         {
//             width = cam.targetTexture.width;
//             height = cam.targetTexture.height;
//         }
//         else
//         {
//             // pixelWidth/pixelHeight 会考虑 viewport rect
//             width = cam.pixelWidth;
//             height = cam.pixelHeight;
//         }

//         rec.frame_idx = frameIdx;
//         rec.time = t;
//         rec.width = width;
//         rec.height = height;
//         rec.label = label;

//         // Safety: keep record joint list aligned with joints[]
//         if (rec.joints2d == null) rec.joints2d = new List<Joint2D>();
//         if (rec.joints2d.Count != joints.Length)
//         {
//             rec.joints2d.Clear();
//             foreach (var j in joints)
//                 rec.joints2d.Add(new Joint2D { name = j != null ? j.name : "null" });
//         }

//         for (int i = 0; i < joints.Length; i++)
//         {
//             if (joints[i] == null) continue;

//             // ✅ viewport coordinates (0~1)
//             Vector3 vp = cam.WorldToViewportPoint(joints[i].position);
//             Joint2D jData = rec.joints2d[i];

//             bool inFront = vp.z > 0f;
//             bool inView = (vp.x >= 0f && vp.x <= 1f && vp.y >= 0f && vp.y <= 1f);

//             jData.conf = (inFront && inView) ? 1f : 0f;

//             float x = vp.x;
//             float y = vp.y;

//             // viewport 默认 (0,0)=BottomLeft -> 你要 TopLeft 就翻一下
//             if (flipYToTopLeft) y = 1f - y;

//             // ⚠️ viewport 下通常不要再做 recorderFlipVerticalOn，否则会二次翻转
//             // 如果你确实需要（比如你后续保存的图像又翻了一次），再打开这行：
//             // if (recorderFlipVerticalOn) y = 1f - y;

//             jData.x = x;
//             jData.y = y;
//         }

//         w.WriteLine(JsonUtility.ToJson(rec));
//     }


//     void GetAllChildren(Transform parent, List<Transform> result)
//     {
//         foreach (Transform child in parent)
//         {
//             result.Add(child);
//             GetAllChildren(child, result);
//         }
//     }

//     void OnDestroy() => CloseAll();

//     void CloseAll()
//     {
//         if (wLeft != null) { wLeft.Flush(); wLeft.Close(); wLeft = null; }
//         if (wRight != null) { wRight.Flush(); wRight.Close(); wRight = null; }
//     }
// }
