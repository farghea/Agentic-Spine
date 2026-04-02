from pydantic import BaseModel, Field, field_validator
from typing import Literal, Annotated

class SubjectFilter(BaseModel):
    # Literal ensures only these three strings are accepted
    sex: Literal["male", "female", "any"] = "any"
    
    # Each filter is a numeric list sorted increasingly (single value or many).
    age_range: list[Annotated[float, Field(ge=0)]] = Field(min_length=1)
    weight_range: list[Annotated[float, Field(ge=0)]] = Field(min_length=1)
    height_range: list[Annotated[float, Field(ge=0)]] = Field(min_length=1)

    @field_validator("age_range", "weight_range", "height_range")
    @classmethod
    def sort_ranges_increasingly(cls, value: list[float]) -> list[float]:
        return sorted(value)
    

class AnalysisResult(BaseModel):
    is_relevant: bool
    subject_filter: SubjectFilter
    activity_keys: list[str] = Field(
        min_length=1,
        description="List of activity keys that are relevant to the subject filter."
    )
    verification: str = Field(description="A short status message for the user")

    @field_validator("activity_keys")
    @classmethod
    def validate_activity_keys(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if not cleaned:
            raise ValueError("activity_keys must be a non-empty list.")
        return cleaned


class RequestSplitResult(BaseModel):
    simulation_request: str = Field(
        min_length=1,
        description="Prompt fragment containing subject/activity setup for simulation."
    )
    analysis_request: str = Field(
        default="",
        description="Optional post-simulation analysis question."
    )

    @field_validator("simulation_request", mode="before")
    @classmethod
    def normalize_simulation_request(cls, value) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("simulation_request must be a non-empty string.")
        return text

    @field_validator("analysis_request", mode="before")
    @classmethod
    def normalize_analysis_request(cls, value) -> str:
        return str(value or "").strip()

if __name__ == "__main__":

    #--- Example Usage ---

    # 1. Valid Data
    valid_data = {
        "sex": "male",
        "age_range": [18, 65],
        "weight_range": [50.5, 120.0],
        "height_range": [150, 200]
    }
    filter_obj = SubjectFilter(**valid_data)
    print("✅ Valid Filter:", filter_obj.model_dump())

    # 2. Unsorted values are accepted and normalized increasingly
    unsorted_data = {
        "sex": "any",
        "age_range": [80, 20],
        "weight_range": [100, 70],
        "height_range": [180, 160]
    }
    normalized_filter = SubjectFilter(**unsorted_data)
    print("✅ Normalized Filter:", normalized_filter.model_dump())

    # 3. Population variation: multiple age targets with shared height/weight
    population_data = {
        "sex": "female",
        "age_range": [70, 60, 50],
        "weight_range": [65],
        "height_range": [1.62]
    }
    population_filter = SubjectFilter(**population_data)
    print("✅ Population Filter:", population_filter.model_dump())