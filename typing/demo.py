from typing import NewType

UserId = NewType('UserId', int)
some_id = UserId(-1)
print(some_id)