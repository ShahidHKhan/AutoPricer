import modal

app = modal.App("vectorstore-setup")
vol = modal.Volume.from_name("cars-vectorstore-vol")


@app.function(volumes={"/vol": vol})
def unzip_vectorstore():
    import zipfile
    import os

    # Clean up the flat, backslash-named files from the failed attempt
    for name in os.listdir("/vol"):
        if name.startswith("cars_vectorstore\\"):
            os.remove(os.path.join("/vol", name))

    with zipfile.ZipFile("/vol/cars_vectorstore.zip") as z:
        for info in z.infolist():
            # Windows zip entries use backslashes; normalize to real paths
            normalized_name = info.filename.replace("\\", "/")
            dest_path = os.path.join("/vol", normalized_name)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with z.open(info) as source, open(dest_path, "wb") as target:
                target.write(source.read())

    vol.commit()
    print("Contents of /vol after extraction:", os.listdir("/vol"))
    print("Contents of /vol/cars_vectorstore:", os.listdir("/vol/cars_vectorstore"))