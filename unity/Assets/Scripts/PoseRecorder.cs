using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public class Joint3D
{
    public string name;
    public float x, y, z;
}

[Serializable]
public class AnimInfo
{
    public string animatorName;
    public int layer;
    public int stateShortHash;
    public float normalizedTime;
    public string stateName; // animation label
}

[Serializable]
public class FrameRecord
{
    public int frame;
    public float time;
    public List<Joint3D> joints3d = new List<Joint3D>();
    public AnimInfo anim;
}

public class PoseRecorder : MonoBehaviour
{
    [Header("Bone Scan Settings")]
    [Tooltip("rootBone以下のSkinnedMeshRendererからボーンを自動取得します")]
    public bool autoScanFromMesh = true;
    public Transform rootBone;

    [Header("Manual Setup (if auto scan is off)")]
    public Transform[] joints;

    [Header("Animator Settings")]
    public Animator animator;
    public int animatorLayer = 0;

    [Tooltip("每隔多少帧采样一次（1=每帧）")]
    public int poseEveryNFrames = 1;

    [Header("Dataset Output")]
    public string outRootFolder = "SkiDataset";
    public string subjectId = "S001";
    public string actionId = "A001";
    public string takeId = "take_0001";
    public string kpt3dFileName = "kpt3d_poseRecorder.npy";

    [Header("Optional JSONL")]
    public bool exportJsonl = false;
    public string jsonlFileName = "pose3d.jsonl";

    public bool recordOnPlay = true;

    private StreamWriter writer;
    private FrameRecord reusableRecord = new FrameRecord();
    private List<float> kpt3dBuffer = new List<float>();
    private int sampledFrames = 0;

    void Start()
    {
        Application.targetFrameRate = 30; // 録画時は高フレームレートを目指す
        if (!recordOnPlay) return;

        // 1. ボーンの自動スキャン
        if (autoScanFromMesh && rootBone != null)
        {
            var list = new List<Transform>();

            // キャラクターのメッシュに紐付いた「実際のボーン」を取得
            var smr = rootBone.GetComponentInChildren<SkinnedMeshRenderer>();
            if (smr != null && smr.bones.Length > 0)
            {
                list.AddRange(smr.bones);
            }
            else
            {
                // メッシュがない場合は全階層（自分自身を含む）
                list.Add(rootBone);
                GetAllChildren(rootBone, list);
            }
            joints = list.ToArray();
            Debug.Log($"[PoseRecorder] {joints.Length} 個の関節を登録しました。");
        }

        // 2. 各種コンポーネントの自動取得
        if (animator == null) animator = GetComponentInChildren<Animator>();

        // 3. 保存先のセットアップ
        SetupOutput();

        // 4. メモリ効率のためのデータ構造準備
        PrepareReusableList();
    }

    void SetupOutput()
    {
        if (!exportJsonl) return;

        string takeRoot = Path.Combine(Application.dataPath, "..", outRootFolder, "subjects", subjectId, "actions", actionId, "takes", takeId);
        string kpt3dDir = Path.Combine(takeRoot, "kpt3d");
        Directory.CreateDirectory(kpt3dDir);

        string outPath = Path.Combine(kpt3dDir, jsonlFileName);
        writer = new StreamWriter(outPath, false, Encoding.UTF8);
    }

    void PrepareReusableList()
    {
        if (joints == null) return;
        reusableRecord.joints3d.Clear();
        for (int i = 0; i < joints.Length; i++)
        {
            reusableRecord.joints3d.Add(new Joint3D());
        }
    }

    void LateUpdate()
    {
        if (joints == null || joints.Length == 0) return;

        if (poseEveryNFrames > 1 && (Time.frameCount % poseEveryNFrames != 0)) return;

        reusableRecord.frame = sampledFrames;
        reusableRecord.time = Time.time;

        // ボーン座標の更新
        for (int i = 0; i < joints.Length; i++)
        {
            if (joints[i] == null) continue;
            var jData = reusableRecord.joints3d[i];
            jData.name = joints[i].name;
            jData.x = joints[i].position.x;
            jData.y = joints[i].position.y;
            jData.z = joints[i].position.z;

            kpt3dBuffer.Add(jData.x);
            kpt3dBuffer.Add(jData.y);
            kpt3dBuffer.Add(jData.z);
        }

        sampledFrames++;

        // アニメーション情報の取得
        if (animator != null)
        {
            var st = animator.GetCurrentAnimatorStateInfo(animatorLayer);
            if (reusableRecord.anim == null)
                reusableRecord.anim = new AnimInfo();

            reusableRecord.anim.animatorName = animator.name;
            reusableRecord.anim.layer = animatorLayer;
            reusableRecord.anim.stateShortHash = st.shortNameHash;
            reusableRecord.anim.normalizedTime = st.normalizedTime;

            // ✅ animation label（最关键）
            var clips = animator.GetCurrentAnimatorClipInfo(animatorLayer);
            if (clips != null && clips.Length > 0 && clips[0].clip != null)
            {
                reusableRecord.anim.stateName = clips[0].clip.name;
            }
            else
            {
                reusableRecord.anim.stateName = "Unknown";
            }

        }

        if (writer != null)
            writer.WriteLine(JsonUtility.ToJson(reusableRecord));
    }

    void GetAllChildren(Transform parent, List<Transform> result)
    {
        foreach (Transform child in parent)
        {
            result.Add(child);
            GetAllChildren(child, result);
        }
    }

    void OnDestroy() => Close();
    void OnApplicationQuit() => Close();

    // void Close()
    // {
    //     if (writer != null)
    //     {
    //         writer.Flush();
    //         writer.Close();
    //         writer = null;
    //         Debug.Log("[PoseRecorder] 録画を終了し、保存しました。");
    //     }
    // }
    public void Close()
    {
        WriteKpt3DNpy();

        if (writer != null)
        {
            writer.Flush();
            writer.Close();
            writer = null;
        }

        Debug.Log($"[PoseRecorder] 录制结束，3D关键点已保存（T={sampledFrames}, J={(joints != null ? joints.Length : 0)}）。");
    }

    void WriteKpt3DNpy()
    {
        if (sampledFrames <= 0 || joints == null || joints.Length == 0) return;

        string takeRoot = Path.Combine(Application.dataPath, "..", outRootFolder, "subjects", subjectId, "actions", actionId, "takes", takeId);
        string kpt3dDir = Path.Combine(takeRoot, "kpt3d");
        Directory.CreateDirectory(kpt3dDir);

        string path = Path.Combine(kpt3dDir, kpt3dFileName);
        WriteFloatNpy(path, kpt3dBuffer, sampledFrames, joints.Length, 3);
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

}