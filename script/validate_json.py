import json

def validate_json(json_string):
    try:
        json.loads(json_string)
        print("✅ JSON valide.\n")
    except json.JSONDecodeError as e:
        print(f"❌ Erreur JSON : {e}")

# Exemple d'utilisation
with open("odsl/example-context.json", "r") as f:
    contenu = f.read()
    validate_json(contenu)

    # Maintenant pour l'afficher joliment :
    print(json.dumps(json.loads(contenu), indent=2))
