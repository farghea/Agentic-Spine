from pydantic import BaseModel, Field, ValidationError, model_validator
from typing import Literal, Tuple, Annotated

class SubjectFilter(BaseModel):
    # Literal ensures only these three strings are accepted
    sex: Literal["male", "female", "any"]
    
    # Using Tuple[float, float] represents the [min, max] structure
    # We use Field to ensure the numbers are physically possible (e.g., non-negative)
    age_range: Tuple[Annotated[float, Field(ge=0)], Annotated[float, Field(ge=0)]]
    weight_range: Tuple[Annotated[float, Field(ge=0)], Annotated[float, Field(ge=0)]]
    height_range: Tuple[Annotated[float, Field(ge=0)], Annotated[float, Field(ge=0)]]

    @model_validator(mode='after')
    def validate_ranges(self) -> 'SubjectFilter':
        """Ensures that in all [min, max] pairs, min <= max."""
        ranges_to_check = {
            "age": self.age_range,
            "weight": self.weight_range,
            "height": self.height_range
        }
        
        for name, r in ranges_to_check.items():
            if r[0] > r[1]:
                raise ValueError(f"{name}_range min ({r[0]}) cannot be greater than max ({r[1]})")
        return self
    

class AnalysisResult(BaseModel):
    is_relevant: bool
    subject_filter: SubjectFilter
    activity_keys: list[str] = Field(description="List of activity keys that are relevant to the subject filter.")
    verification: str = Field(description="A short status message for the user")

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

    # 2. This will FAIL because min > max
    invalid_data = {
        "sex": "any",
        "age_range": [80, 20], # Logical error
        "weight_range": [70, 100],
        "height_range": [160, 180]
    }

    try:
        SubjectFilter(**invalid_data)
    except Exception as e:
        print(f"\n❌ Validation Error: {e}")