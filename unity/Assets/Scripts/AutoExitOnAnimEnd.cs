using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class AutoExitOnAnimEnd : MonoBehaviour
{
    [Header("Animator")]
    public Animator animator;
    public int layer = 0;

    [Tooltip("Optional. If set, only exit when this state finishes (exact state name in Animator).")]
    public string stateName = "";

    [Header("Behavior")]
    public bool stopTimeScaleOnFinish = false;  // 可选：结束后冻结画面
    public bool exitPlayModeInEditor = true;    // 编辑器模式：自动退出Play
    public bool quitAppInBuild = false;         // 打包运行：自动退出程序
    public int extraFramesAfterFinish = 0;      // 播完后再多录N帧（避免最后一帧没写到）

    [Header("Recorder Finalize")]
    [Tooltip("结束前主动调用 PoseRecorder.Close()，确保数据落盘")]
    public bool closePoseRecordersBeforeExit = true;

    [Tooltip("自动查找场景里的 PoseRecorder")]
    public bool autoFindPoseRecorders = true;

    public PoseRecorder[] poseRecorders;

    [Header("Wait Capture")]
    [Tooltip("退出前等待场景内 OneCameraCaptureFrame 全部完成")]
    public bool waitAllCapturesDone = true;

    [Tooltip("自动查找场景里的 OneCameraCaptureFrame")]
    public bool autoFindCaptureFrames = true;

    public OneCameraCaptureFrame[] captureFrames;

    bool finished = false;
    int extraCount = 0;
    int targetHash = 0;
    bool recorderClosed = false;

    void Start()
    {
        if (animator == null)
            animator = GetComponentInChildren<Animator>();

        if (animator == null)
        {
            Debug.LogError("[AutoExitOnAnimEnd] Animator not found.");
            enabled = false;
            return;
        }

        if (!string.IsNullOrEmpty(stateName))
            targetHash = Animator.StringToHash(stateName);

        if (autoFindPoseRecorders)
            poseRecorders = FindObjectsOfType<PoseRecorder>(true);

        if (autoFindCaptureFrames)
            captureFrames = FindObjectsOfType<OneCameraCaptureFrame>(true);
    }

    void Update()
    {
        if (finished) return;

        var st = animator.GetCurrentAnimatorStateInfo(layer);

        // 如果指定了 stateName，就只在该 state 播完时退出
        if (targetHash != 0 && st.shortNameHash != targetHash)
            return;

        // normalizedTime >= 1 表示至少播完一次；同时确保不在 transition 中
        if (st.normalizedTime >= 1.0f && !animator.IsInTransition(layer))
        {
            finished = true;
            extraCount = extraFramesAfterFinish;

            if (stopTimeScaleOnFinish)
                Time.timeScale = 0f;
        }
    }

    void LateUpdate()
    {
        // 用 LateUpdate 计 extraFrames，保证你的 kpt recorder 也在同一帧完成写入
        if (!finished) return;

        if (extraCount > 0)
        {
            extraCount--;
            return;
        }

        if (waitAllCapturesDone && !AreAllCapturesDone())
            return;

        ExitNow();
    }

    void ExitNow()
    {
        CloseRecordersIfNeeded();

#if UNITY_EDITOR
        if (exitPlayModeInEditor)
        {
            Debug.Log("[AutoExitOnAnimEnd] Exit Play Mode.");
            EditorApplication.isPlaying = false;
            return;
        }
#endif
        if (quitAppInBuild)
        {
            Debug.Log("[AutoExitOnAnimEnd] Quit Application.");
            Application.Quit();
        }
    }

    void CloseRecordersIfNeeded()
    {
        if (!closePoseRecordersBeforeExit || recorderClosed) return;

        if (poseRecorders == null || poseRecorders.Length == 0)
            poseRecorders = FindObjectsOfType<PoseRecorder>(true);

        for (int i = 0; i < poseRecorders.Length; i++)
        {
            if (poseRecorders[i] == null) continue;
            poseRecorders[i].Close();
        }

        recorderClosed = true;
    }

    bool AreAllCapturesDone()
    {
        if (captureFrames == null || captureFrames.Length == 0)
        {
            if (autoFindCaptureFrames)
                captureFrames = FindObjectsOfType<OneCameraCaptureFrame>(true);
        }

        if (captureFrames == null || captureFrames.Length == 0)
            return true;

        for (int i = 0; i < captureFrames.Length; i++)
        {
            var cap = captureFrames[i];
            if (cap == null) continue;
            if (!cap.autoRunOnPlay) continue;

            if (!cap.IsCaptureDone)
                return false;
        }

        return true;
    }
}
