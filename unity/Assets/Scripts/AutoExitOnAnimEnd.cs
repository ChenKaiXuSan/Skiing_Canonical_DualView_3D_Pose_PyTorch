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
    public bool stopTimeScaleOnFinish = false;
    public bool exitPlayModeInEditor = true;
    public bool quitAppInBuild = false;
    public int extraFramesAfterFinish = 0;

    [Header("Wait Capture")]
    [Tooltip("退出前等待场景内 OneCameraCaptureFrame 全部完成")]
    public bool waitAllCapturesDone = true;

    [Tooltip("自动查找场景里的 OneCameraCaptureFrame")]
    public bool autoFindCaptureFrames = true;

    public OneCameraCaptureFrame[] captureFrames;

    bool finished;
    int extraCount;
    int targetHash;

    void Start()
    {
        animator ??= GetComponentInChildren<Animator>();

        if (animator == null)
        {
            Debug.LogError("[AutoExitOnAnimEnd] Animator not found.");
            enabled = false;
            return;
        }

        if (!string.IsNullOrEmpty(stateName))
            targetHash = Animator.StringToHash(stateName);

        RefreshCaptureFrames();
    }

    void Update()
    {
        if (finished) return;

        var st = animator.GetCurrentAnimatorStateInfo(layer);
        if (targetHash != 0 && st.shortNameHash != targetHash)
            return;

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

    bool AreAllCapturesDone()
    {
        if (captureFrames == null || captureFrames.Length == 0)
            RefreshCaptureFrames();

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

    void RefreshCaptureFrames()
    {
        if (!autoFindCaptureFrames) return;
        captureFrames = FindObjectsOfType<OneCameraCaptureFrame>(true);
    }
}
