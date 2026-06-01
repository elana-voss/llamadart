/// The kind of compute device the native backend enumerated.
enum GpuDeviceType {
  /// Host CPU.
  cpu,

  /// Dedicated GPU with its own VRAM.
  discreteGpu,

  /// Integrated GPU sharing system memory.
  integratedGpu,

  /// Dedicated accelerator (e.g. NPU).
  accelerator,

  /// Meta/virtual device that aggregates others.
  meta,

  /// Unrecognized device type.
  unknown,
}

/// One device reported by the loaded backends (CPU or a GPU).
///
/// [index] is the position in the backend's full device list (CPU included),
/// for identification and logging. It is not the `mainGpu` value: `mainGpu`
/// indexes the offload device list, which excludes the CPU device, so the
/// caller derives it from a device's position among the non-CPU entries.
class GpuDeviceInfo {
  /// Position in the backend's full device list (CPU included).
  final int index;

  /// Human-readable device name reported by the driver.
  final String name;

  /// Device category. Integrated GPUs report shared system memory, so callers
  /// preferring dedicated VRAM should favor [GpuDeviceType.discreteGpu].
  final GpuDeviceType type;

  /// Free device memory in bytes at enumeration time.
  final int memoryFreeBytes;

  /// Total device memory in bytes.
  final int memoryTotalBytes;

  /// Creates a [GpuDeviceInfo].
  const GpuDeviceInfo({
    required this.index,
    required this.name,
    required this.type,
    required this.memoryFreeBytes,
    required this.memoryTotalBytes,
  });

  /// A dedicated GPU with its own VRAM (not an integrated/shared-memory GPU).
  bool get isDiscreteGpu => type == GpuDeviceType.discreteGpu;

  @override
  String toString() =>
      'GpuDeviceInfo(index: $index, name: $name, type: ${type.name}, '
      'memoryFree: $memoryFreeBytes, memoryTotal: $memoryTotalBytes)';
}
