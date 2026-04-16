from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)
    role: str = "user"
    admin_code: str | None = None


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class AuthTokenData(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str


class MeData(BaseModel):
    username: str
    role: str
