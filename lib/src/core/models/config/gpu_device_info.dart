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

/// One inference-capable device reported by a GPU backend.
///
/// [index] is the position within that backend's device list and is the value
/// to pass as [ModelParams.mainGpu] to pin offload to this device (with
/// [ModelParams.splitMode] set to none for single-GPU use).
class GpuDeviceInfo {
  /// Position within the backend's device list (the `mainGpu` value).
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
