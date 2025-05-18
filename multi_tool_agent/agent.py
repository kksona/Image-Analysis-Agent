import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent

def get_objects(image_path: str) -> dict:
    """Retrieves the objects present in an image.

    Args:
        image_path (str): The url of the image.

    Returns:
        dict: status and result or error msg.
    """
    if(image_path):
        return {
            "status":"success",
            "report": (f'The image path is {image_path}')
        }
    else:
        return {
            "status": "failure",
            "report": "No image path was provided."
        }


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """

    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": (
                f"Sorry, I don't have timezone information for {city}."
            ),
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    return {"status": "success", "report": report}


root_agent = Agent(
    name="weather_time_agent",
    model="gemini-2.0-flash",
    description=(
        "Agent to answer questions about the time in a city and provides the image path which is given to it."
    ),
    instruction=(
        "You are a helpful agent who can answer user questions about the time in a city and also provide the image path which is given by the user."
    ),
    tools=[get_objects, get_current_time],
)