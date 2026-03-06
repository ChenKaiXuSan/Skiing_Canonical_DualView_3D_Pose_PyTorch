using System;
using System.IO;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CharacterKpt3DExporter : MonoBehaviour
{
    [Header("Output")]
    [Tooltip("Dataset root folder under project root")]
    public string outRootFolder = "SkiDataset";

    [Tooltip("Character folder name under SkiDataset")]
    public string characterFolderName = "female";

    [Tooltip("Use GameObject name when characterFolderName is empty")]
    public bool autoUseTargetNameAsCharacterFolder = true;

    [Tooltip("Overwrite existing 3D kpt npy files")]
    public bool overwriteExistingKpt3d = false;

    [Tooltip("Save per-frame 3D keypoints as frame_XXXXXX.npy")]
    public bool exportPerFrameKpt3d = true;

    [Tooltip("Also save merged clip tensor to kpt3d.npy with shape (T,J,3)")]
    public bool exportMergedKpt3d = false;

    [Tooltip("Per-frame npy filename prefix")]
    public string perFrameFilePrefix = "frame";

    [Header("Run")]
    public bool autoRunOnPlay = true;
    public float startDelaySec = 0f;

    [Header("Target")]
    public Animator targetAnimator;

    [Tooltip("Auto scan joints from rootBone")]
    public bool autoScanAllChildren = true;

    public Transform rootBone;
    public Transform[] joints;

    [Header("Sampling")]
    [Tooltip("Sample every N clip frames")]
    public int poseEveryNFrames = 1;

    [Tooltip("Append final frame if not aligned with stride")]
    public bool includeLastFrame = true;

    [Tooltip("Log per-clip summary")]
    public bool logClipSummary = true;

    public bool IsExportStarted { get; private set; }
    public bool IsExportDone { get; private set; }

    IEnumerator Start()
    {
        IsExportStarted = true;
        IsExportDone = false;

        if (!autoRunOnPlay)
            yield break;

        if (startDelaySec > 0f)
            yield return new WaitForSeconds(startDelaySec);

        if (targetAnimator == null)
            targetAnimator = GetComponent<Animator>();

        if (targetAnimator == null)
        {
            Debug.LogError("[CharacterKpt3DExporter] targetAnimator is null.");
            yield break;
        }

        if (autoScanAllChildren && rootBone != null)
            ResolveJointsFromRoot();

        if (joints == null || joints.Length == 0)
        {
            Debug.LogError("[CharacterKpt3DExporter] joints is empty.");
            yield break;
        }

        var ac = targetAnimator.runtimeAnimatorController;
        if (ac == null)
        {
            Debug.LogError("[CharacterKpt3DExporter] RuntimeAnimatorController is null.");
            yield break;
        }

        string resolvedCharacter = ResolveCharacterFolder();
        string datasetRoot = Path.Combine(Application.dataPath, "..", "..", outRootFolder);
        string characterRoot = Path.Combine(datasetRoot, resolvedCharacter);
        Directory.CreateDirectory(characterRoot);

        var clips = GetUniqueControllerClips(ac);
        if (clips.Count == 0)
        {
            Debug.LogError("[CharacterKpt3DExporter] No clips found in RuntimeAnimatorController.");
            yield break;
        }

        float oldSpeed = targetAnimator.speed;
        bool oldEnabled = targetAnimator.enabled;

        try
        {
            for (int i = 0; i < clips.Count; i++)
            {
                yield return StartCoroutine(ExportClipKpt3D(characterRoot, clips[i], i, clips.Count));
            }
        }
        finally
        {
            targetAnimator.speed = oldSpeed;
            targetAnimator.enabled = oldEnabled;
            IsExportDone = true;
        }

        Debug.Log("[CharacterKpt3DExporter] Export finished.");
    }

    void ResolveJointsFromRoot()
    {
        var list = new List<Transform>();
        var smr = rootBone.GetComponentInChildren<SkinnedMeshRenderer>();
        if (smr != null && smr.bones != null && smr.bones.Length > 0)
        {
            list.AddRange(smr.bones);
        }
        else
        {
            GetAllChildren(rootBone, list);
        }

        var dedup = new List<Transform>();
        var seen = new HashSet<Transform>();
        for (int i = 0; i < list.Count; i++)
        {
            var t = list[i];
            if (t != null && seen.Add(t))
                dedup.Add(t);
        }

        joints = dedup.ToArray();
    }

    IEnumerator ExportClipKpt3D(string characterRoot, AnimationClip clip, int clipIndex, int clipCount)
    {
        if (clip == null)
            yield break;

        string safeActionName = MakeSafePathName(clip.name);
        string kpt3dDir = Path.Combine(characterRoot, safeActionName, "kpt3d");
        Directory.CreateDirectory(kpt3dDir);

        string kpt3dPath = Path.Combine(kpt3dDir, "kpt3d.npy");

        int frameCount = Mathf.Max(1, Mathf.RoundToInt(clip.length * clip.frameRate));
        int stride = Mathf.Max(1, poseEveryNFrames);

        var sampleFrames = new List<int>(frameCount / stride + 2);
        for (int f = 0; f < frameCount; f += stride)
            sampleFrames.Add(f);

        int lastFrame = Mathf.Max(0, frameCount - 1);
        if (includeLastFrame && (sampleFrames.Count == 0 || sampleFrames[sampleFrames.Count - 1] != lastFrame))
            sampleFrames.Add(lastFrame);

        var mergedBuffer = exportMergedKpt3d ? new List<float>(sampleFrames.Count * joints.Length * 3) : null;
        int savedPerFrameCount = 0;

        for (int s = 0; s < sampleFrames.Count; s++)
        {
            int localFrame = sampleFrames[s];
            float denom = Mathf.Max(1f, frameCount - 1f);
            float t01 = Mathf.Clamp01(localFrame / denom);
            float tSec = t01 * clip.length;

            targetAnimator.enabled = false;
            clip.SampleAnimation(targetAnimator.gameObject, tSec);

            var frameBuffer = new List<float>(joints.Length * 3);
            AppendKpt3DWorldTJC3(frameBuffer);

            if (exportPerFrameKpt3d)
            {
                string frameName = string.IsNullOrWhiteSpace(perFrameFilePrefix) ? "frame" : perFrameFilePrefix;
                string framePath = Path.Combine(kpt3dDir, $"{frameName}_{s:D6}.npy");
                if (!File.Exists(framePath) || overwriteExistingKpt3d)
                {
                    WriteFloatNpy(framePath, frameBuffer, joints.Length, 3);
                    savedPerFrameCount++;
                }
            }

            if (mergedBuffer != null)
                mergedBuffer.AddRange(frameBuffer);

            // Avoid freezing editor on very long clips.
            if ((s & 63) == 0)
                yield return null;
        }

        if (mergedBuffer != null && (!File.Exists(kpt3dPath) || overwriteExistingKpt3d))
            WriteFloatNpy(kpt3dPath, mergedBuffer, sampleFrames.Count, joints.Length, 3);

        if (logClipSummary)
        {
            Debug.Log(
                $"[CharacterKpt3DExporter] [{clipIndex + 1}/{clipCount}] action={safeActionName}, " +
                $"clipLen={clip.length:F3}s, clipFps={clip.frameRate:F2}, clipFrames={frameCount}, " +
                $"sampledFrames={sampleFrames.Count}, joints={joints.Length}, " +
                $"savedPerFrame={savedPerFrameCount}, merged={(exportMergedKpt3d ? kpt3dPath : "disabled")}");
        }
    }

    void AppendKpt3DWorldTJC3(List<float> outBuffer)
    {
        for (int i = 0; i < joints.Length; i++)
        {
            var t = joints[i];
            Vector3 p = t != null ? t.position : Vector3.zero;
            outBuffer.Add(p.x);
            outBuffer.Add(p.y);
            outBuffer.Add(p.z);
        }
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

    void WriteFloatNpy(string path, List<float> data, int d0, int d1)
    {
        using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write))
        using (var bw = new BinaryWriter(fs))
        {
            bw.Write((byte)0x93);
            bw.Write(Encoding.ASCII.GetBytes("NUMPY"));
            bw.Write((byte)1);
            bw.Write((byte)0);

            string dict = $"{{'descr': '<f4', 'fortran_order': False, 'shape': ({d0}, {d1}), }}";
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

    List<AnimationClip> GetUniqueControllerClips(RuntimeAnimatorController ac)
    {
        var result = new List<AnimationClip>();
        if (ac == null || ac.animationClips == null)
            return result;

        var seen = new HashSet<AnimationClip>();
        for (int i = 0; i < ac.animationClips.Length; i++)
        {
            var clip = ac.animationClips[i];
            if (clip == null)
                continue;
            if (seen.Add(clip))
                result.Add(clip);
        }

        return result;
    }

    string ResolveCharacterFolder()
    {
        if (!string.IsNullOrWhiteSpace(characterFolderName))
            return MakeSafePathName(characterFolderName);
        if (autoUseTargetNameAsCharacterFolder)
            return MakeSafePathName(gameObject.name);
        return "UnknownCharacter";
    }

    string MakeSafePathName(string name)
    {
        string safe = string.IsNullOrWhiteSpace(name) ? "UnknownAction" : name;
        char[] invalid = Path.GetInvalidFileNameChars();
        for (int i = 0; i < invalid.Length; i++)
            safe = safe.Replace(invalid[i], '_');
        return safe;
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
