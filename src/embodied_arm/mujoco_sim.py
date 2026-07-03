from __future__ import annotations


def check_mujoco_runtime() -> dict[str, str | bool]:
    try:
        import mujoco
    except Exception as exc:
        return {"simulation_ok": False, "error": str(exc)}
    return {"simulation_ok": True, "mujoco_version": getattr(mujoco, "__version__", "unknown")}

