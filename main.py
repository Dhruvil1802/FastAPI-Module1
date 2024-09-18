from datetime import datetime, time, timedelta
from enum import Enum
from typing import Literal, Union
from uuid import UUID

from fastapi import (
    Body,
    FastAPI,
    Query,
    Path,
    Cookie,
    Header,
    status,
    Form,
    File,
    UploadFile,
    HTTPException,
    Request,
)
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, HttpUrl, EmailStr
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import HTMLResponse

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "hello "}

@app.post("/")
async def create_post():
    return {"message": "hello post "}

@app.put("/")
async def update_put():
    return {"message": "hello put "}

@app.get("/users")
async def get_all_users():
    return {"message": "list users route"}

@app.get("/users/me")
async def get_logged_in_user():
    return {"Message": "this is the current user"}

@app.get("/users/{user_id}")
async def get_specific_user(user_id: str):
    return {"user_id": user_id}

class CategoryEnum(str, Enum):
    fruits = "fruits"
    vegetables = "vegetables"
    dairy = "dairy"

@app.get("/categories/{category_name}")
async def get_category(category_name: CategoryEnum):
    if category_name == CategoryEnum.vegetables:
        return {"category_name": category_name, "message": "you are healthy"}

    if category_name.value == "fruits":
        return {
            "category_name": category_name,
            "message": "you are still healthy",
        }
    return {"category_name": category_name, "message": "i like chocolate milk"}

dummy_database = [{"item_title": "Foo"}, {"item_title": "Bar"}, {"item_title": "Baz"}]

@app.get("/old_data")
async def fetch_old_data(skip: int = 0, limit: int = 10):
    return dummy_database[skip : skip + limit]

@app.get("/items/{item_id}")
async def fetch_item(
    item_id: str, query_param: str, q: str | None = None, is_short: bool = False
):
    item = {"item_id": item_id, "query_param": query_param}
    if q:
        item.update({"q": q})
    if not is_short:
        item.update(
            {
                "description": " update description of an item"
            }
        )
    return item

@app.get("/users/{user_id}/items/{item_id}")
async def fetch_user_item(
    user_id: int, item_id: str, q: str | None = None, is_short: bool = False
):
    user_item = {"item_id": item_id, "owner_id": user_id}
    if q:
        user_item.update({"q": q})
    if not is_short:
        user_item.update(
            {
                "description": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ut consectetur."
            }
        )
        return user_item

class Product(BaseModel):
    title: str
    description: str | None = None
    cost: float
    tax_amount: float | None = None

@app.post("/items")
async def add_item(product: Product):
    product_dict = product.dict()
    if product.tax_amount:
        total_price = product.cost + product.tax_amount
        product_dict.update({"total_price": total_price})
    return product_dict

@app.put("/items/{item_id}")
async def update_item_with_put(item_id: int, product: Product, q: str | None = None):
    response = {"item_id": item_id, **product.dict()}
    if q:
        response.update({"q": q})
    return response

@app.get("/search_items")
async def search_items(
    query: str | None = Query(
        None,
        min_length=3,
        max_length=10,
        title="Query string",
        description="A sample query string.",
        alias="search-query",
    )
):
    result_set = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if query:
        result_set.update({"query": query})
    return result_set


@app.get("/items_hidden")
async def hidden_query_route(
    hidden_query: str | None = Query(None, include_in_schema=False)
):
    if hidden_query:
        return {"hidden_query": hidden_query}
    return {"hidden_query": "Not found"}

@app.get("/items_validation/{item_id}")
async def read_items_validation(
    *,
    #item_id must be path parameter
    item_id: int = Path(..., title="The ID of the item to get", gt=10, le=100),
    q: str = "hello",
    #size must be query parameter
    size: float = Query(..., gt=0, lt=7.75)
):
    
    results = {"item_id": item_id, "size": size}
    if q:
        results.update({"q": q})
    return results


class Item(BaseModel):
    name: str
    description: str | None = Field(
        None, title="The description of the item", max_length=300
    )
    price: float = Field(..., gt=0, description="The price must be greater than zero.")
    tax: float | None = None

#passed request body in query parameter
@app.put("/items/{item_id}")
async def update_item1(item_id: int, item: Item = Body(..., embed=True)):
    results = {"item_id": item_id, "item": item}
    return results

class Image(BaseModel):
    url: HttpUrl
    name: str


class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None
    tags: set[str] = []
    image: list[Image] | None = None


class Offer(BaseModel):
    name: str
    description: str | None = None
    price: float
    items: list[Item]


@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    results = {"item_id": item_id, "item": item}
    return results

@app.post("/offers")
async def create_offer(offer: Offer = Body(..., embed=True)):
    return offer


@app.post("/images/multiple")
async def create_multiple_images(images: list[Image]):
    return images


@app.post("/blah")
async def create_some_blahs(blahs: dict[int, float]):
    return blahs

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None



@app.put("/items/{item_id}")
async def update_items2(
    item_id: int,
    item: Item = Body(
        ...,
        examples={
            "normal": {
                "summary": "A normal example",
                "description": "A __normal__ item works _correctly_",
                "value": {
                    "name": "Foo",
                    "description": "A very nice Item",
                    "price": 16.25,
                    "tax": 1.67,
                },
            },          
            "converted": {
                "summary": "An example with converted data",
                "description": "FastAPI can convert price `strings` to actual `numbers` automatically",
                "value": {"name": "Bar", "price": "16.25"},
            },
            "invalid": {
                "summary": "Invalid data is rejected with an error",
                "description": "Hello youtubers",
                "value": {"name": "Baz", "price": "sixteen point two five"},
            },
        },
    ),
):
    results = {"item_id": item_id, "item": item}
    return results


@app.put("/items/{item_id}")
async def read_items(
    item_id: UUID,
    start_date: Optional[datetime] = Body(None),
    end_date: Optional[datetime] = Body(None),
    repeat_at: Optional[time] = Body(None),
    process_after: Optional[timedelta] = Body(None),
):
    if start_date and process_after :
        start_process = start_date + process_after 
    else:
        None
    if start_process and end_date:
        duration = end_date - start_process 
    else:
        None
    return {
        "item_id": item_id,
        "start_date": start_date,
        "end_date": end_date,
        "repeat_at": repeat_at,
        "process_after": process_after,
        "start_process": start_process,
        "duration": duration,
    }

@app.get("/items")
async def read_items(
    cookie_id: str | None = Cookie(None),
    accept_encoding: str | None = Header(None),
    sec_ch_ua: str | None = Header(None),
    user_agent: str | None = Header(None),
    x_token: list[str] | None = Header(None),
):
    return {
        "cookie_id": cookie_id,
        "Accept-Encoding": accept_encoding,
        "sec-ch-ua": sec_ch_ua,
        "User-Agent": user_agent,
        "X-Token values": x_token,
    }

# response_model

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float = 10.5
    tags: list[str] = []


items = {
    "foo": {"name": "Foo", "price": 50.2},
    "bar": {"name": "Bar", "description": "The bartenders", "price": 62, "tax": 20.2},
    "baz": {"name": "Baz", "description": None, "price": 50.2, "tax": 10.5, "tags": []},
}


@app.get("/items/{item_id}", response_model=Item, response_model_exclude_unset=True)
async def read_item(item_id: Literal["foo", "bar", "baz"]):
    return items[item_id]


@app.post("/items/", response_model=Item)
async def create_item(item: Item):
    return item


class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: str | None = None


class UserIn(UserBase):
    password: str


class UserOut(UserBase):
    pass


@app.post("/user/", response_model=UserOut)
async def create_user(user: UserIn):
    return user


@app.get(
    "/items/{item_id}/name",
    response_model=Item,
    response_model_include={"name", "description"},
)
async def read_item_name(item_id: Literal["foo", "bar", "baz"]):
    return items[item_id]


@app.get("/items/{item_id}/public", response_model=Item, response_model_exclude={"tax"})
async def read_items_public_data(item_id: Literal["foo", "bar", "baz"]):
    return items[item_id]

# extra model

class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: str | None = None


class UserIn(UserBase):
    password: str


class UserOut(UserBase):
    pass


class UserInDB(UserBase):
    hashed_password: str


def fake_password_hasher(raw_password: str):
    return f"supersecret{raw_password}"


def fake_save_user(user_in: UserIn):
    hashed_password = fake_password_hasher(user_in.password)
    user_in_db = UserInDB(**user_in.dict(), hashed_password=hashed_password)
    print("User 'saved'.")

    return user_in_db


@app.post("/user/", response_model=UserOut)
async def create_user(user_in: UserIn):
    user_saved = fake_save_user(user_in)
    return user_saved


class BaseItem(BaseModel):
    description: str
    type: str


class CarItem(BaseItem):
    type: Literal["car"] = "car"


class PlaneItem(BaseItem):
    type: Literal["plane"] = "plane"
    size: int


items = {
    "item1": {"description": "All my friends drive a low rider", "type": "car"},
    "item2": {
        "description": "Music is my aeroplane, it's my aeroplane",
        "type": "plane",
        "size": 5,
    },
}


@app.get("/items/{item_id}", response_model=Union[PlaneItem, CarItem])
async def read_item(item_id: Literal["item1", "item2"]):
    return items[item_id]


class ListItem(BaseModel):
    name: str
    description: str


list_items = [
    {"name": "Foo", "description": "There comes my hero"},
    {"name": "Red", "description": "It's my aeroplane"},
]


@app.get("/list_items/", response_model=list[ListItem])
async def read_items():
    return items


@app.get("/arbitrary", response_model=dict[str, float])
async def get_arbitrary():
    return {"foo": 1, "bar": "2"}

# Form

@app.post("/login/")
async def login(username: str = Form(...), password: str = Body(...)):
    print("password", password)
    return {"username": username}


@app.post("/login-json/")
async def login_json(username: str = Body(...), password: str = Body(...)):
    print("password", password)
    return {"username": username}

upload files

@app.post("/files/")
async def create_file(
    files: list[bytes] = File(..., description="A file read as bytes")
):
    return {"file_sizes": [len(file) for file in files]}


@app.post("/uploadfile/")
async def create_upload_file(
    files: list[UploadFile] = File(..., description="A file read as UploadFile")
):
    return {"filename": [file.filename for file in files]}

# error handling

items = {"foo": "The Foo Wrestlers"}


@app.get("/items/{item_id}")
async def read_item(item_id: str):
    if item_id not in items:
        raise HTTPException(
            status_code=404,
            detail="Item not found",
            headers={"X-Error": "There goes my error"},
        )
    return {"item": items[item_id]}


class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name


@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something. There goes a rainbow..."},
    )


@app.get("/unicorns/{name}")
async def read_unicorns(name: str):
    if name == "yolo":
        raise UnicornException(name=name)
    return {"unicorn_name": name}


@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request, exc):
    print(f"OMG! An HTTP error!: {repr(exc)}")
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    print(f"OMG! The client sent invalid data!: {exc}")
    return await request_validation_exception_handler(request, exc)


@app.get("/blah_items/{item_id}")
async def read_items(item_id: int):
    if item_id == 3:
        raise HTTPException(status_code=418, detail="Nope! I don't like 3.")
    return {"item_id": item_id}

