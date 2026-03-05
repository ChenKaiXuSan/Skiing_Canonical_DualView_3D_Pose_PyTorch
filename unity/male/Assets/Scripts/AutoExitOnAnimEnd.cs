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

    [Tooltip("是否启用基于 Animator 状态的退出条件。采集脚本自行按 clip 采样时建议关闭。")]
    public bool useAnimationFinishedCondition = false;

    [Tooltip("如果动画结束条件长期不满足，但采集已全部完成，则允许直接退出")]
    public bool allowExitWhenCapturesDone = true;

    [Tooltip("从启动开始的最大等待秒数，超时后可强制退出（<=0 表示禁用，默认禁用）")]
    public float forceExitTimeoutSec = -1f;

    [Tooltip("即使仍有采集未完成，也允许超时后强制退出。默认关闭，避免中途退出。")]
    public bool forceExitEvenIfCapturesPending = false;

    [Header("Wait Capture")]
    [Tooltip("退出前等待场景内 OneCameraCaptureFrame 全部完成")]
    public bool waitAllCapturesDone = true;

    [Tooltip("自动查找场景里的 OneCameraCaptureFrame")]
    public bool autoFindCaptureFrames = true;

    [Tooltip("打印参与采集数量与未完成数量，便于排查提前退出")]
    public bool logCaptureParticipation = true;

    public OneCameraCaptureFrame[] captureFrames;

    bool finished;
    int extraCount;
    int targetHash;
    float startRealtime;
    bool hasLoggedAnimFinish;
    bool hasLoggedTimeoutBlocked;
    int lastLoggedParticipating = -1;
    int lastLoggedTotal = -1;
    int lastLoggedPending = -1;

    void Start()
    {
        animator ??= GetComponentInChildren<Animator>();
        startRealtime = Time.realtimeSinceStartup;

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

        if (allowExitWhenCapturesDone && waitAllCapturesDone && IsCaptureCompletionSatisfied(requireAtLeastOneCapture: true))
        {
            MarkFinished("captures_done");
            return;
        }

        if (useAnimationFinishedCondition && IsAnimationFinished())
            MarkFinished("animation_finished");
    }

    void MarkFinished(string reason)
    {
        if (finished) return;

        finished = true;
        extraCount = extraFramesAfterFinish;

        if (!hasLoggedAnimFinish)
        {
            Debug.Log($"[AutoExitOnAnimEnd] Finish condition reached: {reason}");
            hasLoggedAnimFinish = true;
        }

        if (stopTimeScaleOnFinish)
            Time.timeScale = 0f;
    }

    void LateUpdate()
    {
        if (!finished)
        {
            if (forceExitTimeoutSec > 0f && Time.realtimeSinceStartup - startRealtime >= forceExitTimeoutSec)
            {
                if (!waitAllCapturesDone || forceExitEvenIfCapturesPending || IsCaptureCompletionSatisfied(requireAtLeastOneCapture: false))
                {
                    Debug.LogWarning("[AutoExitOnAnimEnd] Force exit timeout reached.");
                    ExitNow();
                }
                else if (!hasLoggedTimeoutBlocked)
                {
                    Debug.LogWarning("[AutoExitOnAnimEnd] Timeout reached but captures are still running; keep waiting.");
                    hasLoggedTimeoutBlocked = true;
                }
            }
            return;
        }

        if (extraCount > 0)
        {
            extraCount--;
            return;
        }

        if (waitAllCapturesDone && !IsCaptureCompletionSatisfied(requireAtLeastOneCapture: false))
        {
            if (forceExitTimeoutSec > 0f && Time.realtimeSinceStartup - startRealtime >= forceExitTimeoutSec)
            {
                if (forceExitEvenIfCapturesPending)
                {
                    Debug.LogWarning("[AutoExitOnAnimEnd] Force exit after timeout while waiting captures.");
                    ExitNow();
                }
                else if (!hasLoggedTimeoutBlocked)
                {
                    Debug.LogWarning("[AutoExitOnAnimEnd] Timeout reached but captures are still running; keep waiting.");
                    hasLoggedTimeoutBlocked = true;
                }
            }
            return;
        }

        ExitNow();
    }

    bool IsAnimationFinished()
    {
        if (animator == null) return false;

        var st = animator.GetCurrentAnimatorStateInfo(layer);
        if (targetHash != 0 && st.shortNameHash != targetHash)
            return false;

        return st.normalizedTime >= 1.0f && !animator.IsInTransition(layer);
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

    bool IsCaptureCompletionSatisfied(bool requireAtLeastOneCapture)
    {
        if (captureFrames == null || captureFrames.Length == 0)
            RefreshCaptureFrames();

        int total = captureFrames != null ? captureFrames.Length : 0;
        if (captureFrames == null || captureFrames.Length == 0)
        {
            TryLogCaptureParticipation(0, total, 0);
            return !requireAtLeastOneCapture;
        }

        int participating = 0;
        int pending = 0;
        string firstPendingName = null;

        for (int i = 0; i < captureFrames.Length; i++)
        {
            var cap = captureFrames[i];
            if (cap == null) continue;
            if (!cap.autoRunOnPlay) continue;

            participating++;

            if (!cap.IsCaptureDone)
            {
                pending++;
                if (string.IsNullOrEmpty(firstPendingName))
                    firstPendingName = cap.name;
            }
        }

        TryLogCaptureParticipation(participating, total, pending, firstPendingName);

        if (pending > 0)
            return false;

        if (participating == 0)
            return !requireAtLeastOneCapture;

        return true;
    }

    void TryLogCaptureParticipation(int participating, int total, int pending, string firstPendingName = null)
    {
        if (!logCaptureParticipation) return;
        if (participating == lastLoggedParticipating && total == lastLoggedTotal && pending == lastLoggedPending) return;

        lastLoggedParticipating = participating;
        lastLoggedTotal = total;
        lastLoggedPending = pending;

        string msg = $"[AutoExitOnAnimEnd] capture participation: participating={participating}/{total}, pending={pending}";
        if (!string.IsNullOrEmpty(firstPendingName))
            msg += $", first_pending={firstPendingName}";
        Debug.Log(msg);
    }

    void RefreshCaptureFrames()
    {
        if (!autoFindCaptureFrames) return;
        captureFrames = FindObjectsOfType<OneCameraCaptureFrame>(true);
    }
}
