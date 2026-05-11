import '../../exceptions.dart';
import 'model_download_manager_base.dart';

/// Native placeholder for the package-managed model download manager.
///
/// Real native HTTP download/resume/cache IO is implemented in later tasks.
class DefaultModelDownloadManager extends ThrowingModelDownloadManager {
  /// Creates a native placeholder download manager.
  const DefaultModelDownloadManager();

  @override
  Object unsupported(String operation) => LlamaUnsupportedException(
    'Native model download manager $operation is not implemented yet.',
  );
}
