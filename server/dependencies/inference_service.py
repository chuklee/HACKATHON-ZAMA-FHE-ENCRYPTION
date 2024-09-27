from server.services import InferenceService


def inference_service() -> InferenceService:
    return InferenceService()
