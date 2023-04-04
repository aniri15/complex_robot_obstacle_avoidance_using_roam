from typing import Optional


def show_render(obj, start, stop):
    obj.hide_render = True
    obj.keyframe_insert("hide_render", frame=start)

    obj.hide_render = False
    obj.keyframe_insert("hide_render", frame=stop)


def show_viewport(obj, start, stop):
    obj.hide_viewport = True
    obj.keyframe_insert("hide_viewport", frame=start)
    obj.hide_viewport = False
    obj.keyframe_insert("hide_viewport", frame=stop)


def hide_render(obj, start, stop):
    # # Make whole object disappear
    obj.hide_render = False
    obj.keyframe_insert("hide_render", frame=start)
    obj.hide_render = True
    obj.keyframe_insert("hide_render", frame=stop)


def hide_viewpoert(obj, start: int, stop: int):
    obj.hide_viewport = False
    obj.keyframe_insert("hide_viewport", frame=start)
    obj.hide_viewport = True
    obj.keyframe_insert("hide_viewport", frame=stop)


def make_disappear(obj, start: int, stop: int):
    start = int(start)
    stop = int(stop)

    if hasattr(obj, "object"):
        obj = obj.object

    for mat in obj.data.materials:
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        # bsdf.inputs["Alpha"].default_value = 1.0
        bsdf.inputs["Alpha"].keyframe_insert("default_value", frame=start)
        # bsdf.inputs["Base Color"].default_value[3] = 1.0
        bsdf.inputs["Base Color"].keyframe_insert("default_value", frame=start)

        bsdf.inputs["Alpha"].default_value = 0.0
        bsdf.inputs["Alpha"].keyframe_insert(data_path="default_value", frame=stop)
        bsdf.inputs["Base Color"].default_value[3] = 0.0
        bsdf.inputs["Base Color"].keyframe_insert("default_value", frame=start)

    hide_render(obj, stop, stop + 1)


def make_appear(obj, start: int, stop: int, alpha: float = 1.0):
    start = int(start)
    stop = int(stop)
    if hasattr(obj, "object"):
        obj = obj.object

    for mat in obj.data.materials:
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        # bsdf.inputs["Alpha"].default_value = 0.0
        bsdf.inputs["Alpha"].keyframe_insert("default_value", frame=start)
        # bsdf.inputs["Base Color"].default_value[3] = 0.0
        bsdf.inputs["Base Color"].keyframe_insert("default_value", frame=start)

        bsdf.inputs["Alpha"].default_value = alpha
        bsdf.inputs["Alpha"].keyframe_insert(data_path="default_value", frame=stop)
        bsdf.inputs["Base Color"].default_value[3] = alpha
        bsdf.inputs["Base Color"].keyframe_insert("default_value", frame=start)

    # Hide oject one key frame after
    show_render(obj, start - 1, start)


def move_to(obj, position, start: Optional[int], stop: int) -> None:
    if hasattr(obj, "object"):
        obj = obj.object

    if start is not None:
        obj.keyframe_insert(data_path="location", frame=start)

    obj.location = tuple(position)
    obj.keyframe_insert(data_path="location", frame=stop)
