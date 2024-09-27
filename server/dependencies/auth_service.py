from server.services import AuthService


def auth_service() -> AuthService:
    return AuthService()
