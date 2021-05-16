from pydantic import BaseModel, validator


class RequestDataItem(BaseModel):
    age: int = 40
    sex: int = 1
    cp: int = 0
    trestbps: int = 120
    chol: int = 240
    fbs: int = 0
    restecg: int = 1
    thalach: int = 150
    exang: int = 0
    oldpeak: float = 0
    slope: int = 2
    ca: int = 1
    thal: int = 3

    @validator('age', always=True)
    def validate_age(cls, age: int):
        if age <= 0:
            raise ValueError('The age must be greater than zero')
        return age

    @validator('sex',  always=True)
    def validate_sex(cls, sex: int):
        if sex != 0 and sex != 1:
            raise ValueError('The sex must be 1 or 0')
        return sex

    @validator('cp')
    def validate_cp(cls, cp: int):
        if cp < 0 or cp > 3:
            raise ValueError('The cp must be 0, 1, 2 or 3')
        return cp

    @validator('fbs')
    def validate_fbs(cls, fbs: int):
        if fbs != 0 and fbs != 1:
            raise ValueError('The fbs must be 0 or 1')
        return fbs

    @validator('exang')
    def validate_exang(cls, exang: int):
        if exang != 0 and exang != 1:
            raise ValueError('The exang must be 0 or 1')
        return exang

    @validator('slope')
    def validate_slope(cls, slope: int):
        if slope < 0 or slope > 2:
            raise ValueError('The slope must be 0, 1 or 2')
        return slope
