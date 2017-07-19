import requests
import json
import os

def address2coords(adress, n_results=1):
    LINK = \
        "https://geocode-maps.yandex.ru/1.x/?geocode={adrs}&results={n_res}&format=json"\
            .format(adrs=adress, n_res=n_results)

    response = requests.get(LINK).text
    response = json.loads(response)["response"]["GeoObjectCollection"]["featureMember"][0]["GeoObject"]["Point"]

    return [float(coord) for coord in response["pos"].split(" ")]


def load_map(pos, m_type, fname, z=16):
    LINK = "https://static-maps.yandex.ru/1.x/?ll={lat},{lon}&z={z}&l={type}&size=200,200"\
        .format(lat=pos["lat"], lon=pos["lon"], z=z, type=m_type)

    label = ".jpg"
    if m_type == "map":
        label = ".png"

    print(fname+label)

    with open(fname + label, "wb") as handle:
        response = requests.get(LINK, stream=True)

        if not response.ok:
            print(response)

        for block in response.iter_content(1024):
            if not block:
                break

            handle.write(block)


def get_the_pictures(adress, save_path="pics"):
    if save_path not in os.listdir("."):
        os.mkdir(save_path)

    coords = address2coords(adress)
    coords = {"lat" : coords[0], "lon" : coords[1]}

    load_map(coords, "map", save_path+"/map")
    load_map(coords, "sat", save_path+"/sat")

    #TODO add map generation with nn

print(get_the_pictures("red square"))