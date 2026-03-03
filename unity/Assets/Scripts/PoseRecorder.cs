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

    [Header("Output")]
    public string outputFileName = "pose3d.jsonl";
    public bool recordOnPlay = true;

    private StreamWriter writer;
    private int baseFrameCount;
    private float baseTime;
    private FrameRecord reusableRecord = new FrameRecord();

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
        SetupWriter();

        // 4. メモリ効率のためのデータ構造準備
        PrepareReusableList();
    }

    void SetupWriter()
    {
        string outDir = Path.Combine(Application.dataPath, "..", "RecordingsPose");
        if (!Directory.Exists(outDir)) Directory.CreateDirectory(outDir);

        string outPath = Path.Combine(outDir, outputFileName);
        writer = new StreamWriter(outPath, false, Encoding.UTF8);

        baseFrameCount = Time.frameCount;
        baseTime = Time.time;
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
        if (writer == null || joints == null) return;

        reusableRecord.frame = Time.frameCount - baseFrameCount;
        reusableRecord.time = Time.time - baseTime;

        // ボーン座標の更新
        for (int i = 0; i < joints.Length; i++)
        {
            if (joints[i] == null) continue;
            var jData = reusableRecord.joints3d[i];
            jData.name = joints[i].name;
            jData.x = joints[i].position.x;
            jData.y = joints[i].position.y;
            jData.z = joints[i].position.z;
        }

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
    if (writer != null)
    {
        writer.Flush();
        writer.Close();
        writer = null;
        Debug.Log("[PoseRecorder] 録画を終了し、保存しました。");
    }
}

}