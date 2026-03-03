#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(CameraRingPlacer))]
public class CameraRingPlacerEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        var placer = (CameraRingPlacer)target;
        GUILayout.Space(10);

        if (GUILayout.Button("Create Camera Ring"))
        {
            Undo.RegisterFullObjectHierarchyUndo(placer.gameObject, "Create Camera Ring");
            placer.CreateRing();
            EditorUtility.SetDirty(placer);
        }
    }
}
#endif
