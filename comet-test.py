from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

data = [{"src": "Le chantier est terminé.", "mt": "The construction site is finished.", "ref": "The work is done."}]
score = model.predict(data)
print(score)
