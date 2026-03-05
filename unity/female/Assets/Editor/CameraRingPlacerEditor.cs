using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(CameraRingPlacer))]
public class CameraRingPlacerEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        EditorGUILayout.Space(10);
        EditorGUILayout.HelpBox("点击下方按钮即可生成相机环（等价于 ContextMenu: Create Camera Ring）。", MessageType.Info);

        var placer = (CameraRingPlacer)target;
        Color oldBg = GUI.backgroundColor;
        GUI.backgroundColor = new Color(0.3f, 0.75f, 1f);
        if (GUILayout.Button("生成相机 (Create Camera Ring)", GUILayout.Height(32)))
        {
            int placerId = placer.GetInstanceID();
            EditorApplication.delayCall += () =>
            {
                var delayedPlacer = EditorUtility.InstanceIDToObject(placerId) as CameraRingPlacer;
                if (delayedPlacer == null) return;

                if (delayedPlacer.gameObject == null) return;

                try
                {
                    Undo.RegisterCompleteObjectUndo(delayedPlacer.gameObject, "Create Camera Ring");
                    delayedPlacer.CreateRing();
                    EditorUtility.SetDirty(delayedPlacer);
                }
                catch (System.Exception ex)
                {
                    Debug.LogException(ex);
                }
            };
        }
        GUI.backgroundColor = oldBg;
    }
}
