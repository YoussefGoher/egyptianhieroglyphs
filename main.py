from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
from torchvision import transforms
import base64
from io import BytesIO

app = Flask(__name__)

# # Hieroglyph meanings dictionary
# hieroglyph_meanings = {
#     "Aa15": "Side-symbol (rib, var.)",
#     "Aa26": "Stolist's cloth (?)",
#     "Aa27": "Side-symbol (nḏ variant)",
#     "Aa28": "Builder's tool (qd, 'build')",
#     "A55": "Man lying on back / corpse",
#     "D1": "Head",
#     "D2": "Face (ḥr)",
#     "D10": "Eye of Horus (Wedjat)",
#     "D19": "Mouth (rʿ)",
#     "D21": "Tooth",
#     "D28": "Placenta-shaped basket",
#     "D34": "Eye (simple)",
#     "D35": "Eye with cosmetic line",
#     "D36": "Eye of Horus (decorated)",
#     "D39": "Ploughshare (actually shoulder)",
#     "D4": "Arm (bent)",
#     "D46": "Hand (open)",
#     "D52": "Bread-loaf on head (ḫ)",
#     "D53": "Jar-stand on head",
#     "D54": "Jar-stand (simple)",
#     "D56": "Sun-disc over head (ḥtp-sign)",
#     "D58": "Back-hill determinative",
#     "D60": "Sky over head (pt)",
#     "D62": "Water-pot on head (nw)",
#     "D156": "UNKNOWN (not in Gardiner)",
#     "E1": "Cattle skin on pole (god)",
#     "E9": "Goose",
#     "E17": "Bee",
#     "E23": "Horned viper (f)",
#     "E34": "Lion",
#     "F4": "Horned viper (folded)",
#     "F9": "Quail-chick (w)",
#     "F12": "Horned viper (simple)",
#     "F13": "Scarab beetle (ḫpr)",
#     "F16": "Tadpole / embryo",
#     "F18": "Hare",
#     "F21": "Calf's fore-leg (ḥpš)",
#     "F22": "Bull's fore-leg reversed",
#     "F23": "Fore-leg of ox (khepesh)",
#     "F26": "Fish (generic)",
#     "F29": "Tilapia fish",
#     "F30": "Ox",
#     "F31": "Hippopotamus",
#     "F32": "Elephant",
#     "F34": "Crocodile",
#     "F35": "Tortoise",
#     "F40": "Sparrow (small bird)",
#     "G1": "Vulture (3ʿ)",
#     "G4": "Basket with handle (nb)",
#     "G5": "Comb (ḥḏ)",
#     "G7": "Cup-mouth (ḥtp alt.)",
#     "G10": "Bread-loaf (t)",
#     "G14": "House-facade (sḥ)",
#     "G17": "Owl (m)",
#     "G21": "Sickle (mꜣ)",
#     "G25": "Palace facade (sḥ)",
#     "G26": "False door",
#     "G29": "Column (djed-pillar style)",
#     "G35": "Boat (dpt)",
#     "G36": "Sail (ḥʿtj)",
#     "G37": "Oar",
#     "G39": "Harp",
#     "G40": "Loom",
#     "G43": "Sandal-strap (s)",
#     "G50": "Short sword (ḫpš)",
#     "H6": "Granary (šnw)",
#     "I5": "Was-sceptre of Sobek (crocodile)",
#     "I9": "Horned viper (f)",
#     "I10": "Cobra erect (uraeus)",
#     "L1": "Lion (recumbent)",
#     "M1": "Papyrus plant (wꜣd)",
#     "M3": "Sickle-shaped oar (mꜣ)",
#     "M4": "Sail (with mast)",
#     "M8": "Wall",
#     "M12": "Sycamore tree",
#     "M16": "Hill-country shrub",
#     "M17": "Water ripple (n)",
#     "M18": "Sky (pt) over land",
#     "M20": "Marshland (sxt)",
#     "M23": "Bundle of reeds (ḥ)",
#     "M26": "Papyrus-scroll (ꜥḫ)",
#     "M29": "Bread-cone on offering-table",
#     "M40": "Loaf on mat",
#     "M41": "Jar with handles",
#     "M42": "Basket with lid",
#     "M44": "Sandal",
#     "M195": "UNKNOWN (not in Gardiner)",
#     "N1": "Sky (pt)",
#     "N2": "Sky with sceptre (grḥ, 'night')",
#     "N5": "Sun-disc (rꜥ)",
#     "N14": "Star",
#     "N16": "Cultivated land (ḥꜥt)",
#     "N17": "Variant of N16 (field)",
#     "N18": "Island / sandy bank (ỉw)",
#     "N19": "Double island (ḥr-ꜥḫty)",
#     "N24": "Irrigated district (nome)",
#     "N25": "Hills (foreign land)",
#     "N26": "Mountain (ḏw)",
#     "N29": "Desert slope (q)",
#     "N30": "Mound with bushes",
#     "N31": "Road with shrubs",
#     "N35": "Water ripple triple (n)",
#     "N36": "Canal with banks",
#     "N37": "Pool (rectangle)",
#     "N41": "Well full of water",
#     "O1": "House-plan (pr)",
#     "O4": "Granary",
#     "O11": "Temple-façade",
#     "O28": "Palace",
#     "O29": "Fortified tower",
#     "O31": "Pyramid",
#     "O34": "Tomb-chapel",
#     "O49": "Column-base",
#     "O50": "Throne",
#     "O51": "Chair",
#     "P1": "Boat (dpt)",
#     "P6": "Mooring-post / hammer",
#     "P8": "Drill bow",
#     "P13": "Chisel",
#     "P98": "Loom weight (rare)",
#     "Q1": "Sandal (ḥ)",
#     "Q3": "Basket with handle",
#     "Q7": "Beer-jar on stand",
#     "R4": "Offering-table (ḥtp)",
#     "R8": "Arm holding scepter",
#     "S24": "Scepter (ḥḳꜣ)",
#     "S28": "Ankh (Ꜥnḫ)",
#     "S29": "Djed-pillar (stability)",
#     "S34": "Was-scepter (dominion)",
#     "S42": "Feather of Maat",
#     "T14": "Bread-cone on brazier",
#     "T20": "Beer-jug",
#     "T21": "Wine-jar",
#     "T22": "Meat-leg on hook",
#     "T28": "Incense-burner",
#     "T30": "Ointment-jar",
#     "U1": "Seed-bag / sun-disc on pole",
#     "U7": "Plough",
#     "U15": "Crescent-moon sickle",
#     "U28": "Star (in a circle)",
#     "U33": "Pestle",
#     "U35": "Fire-drill",
#     "V4": "Road (straight rope)",
#     "V6": "Stone-block",
#     "V7": "Mountain-range (rope var.)",
#     "V13": "Water-pot over fire",
#     "V16": "Hill-determinative",
#     "V22": "Desert-sand symbol",
#     "V24": "Cultivated field",
#     "V25": "Garden-plot",
#     "V28": "Tree (generic)",
#     "V30": "Flower (lotus)",
#     "V31": "Papyrus-plant",
#     "W11": "Boat (uya)",
#     "W14": "Oar",
#     "W15": "Sail",
#     "W18": "Anchor",
#     "W19": "Fish-determinative",
#     "W22": "Bird-determinative",
#     "W24": "Bee-determinative",
#     "W25": "Scarab-determinative",
#     "X1": "Round loaf (t)",
#     "X6": "Conical beer-loaf (pꜣt)",
#     "X8": "Conical loaf (di, 'give')",
#     "Y1": "Roll of cloth (s)",
#     "Y2": "Twisted flax (ḥḳ)",
#     "Y3": "Net (ḥ)",
#     "Y5": "Basket with handles (misc.)",
#     "Z1": "Vertical stroke (plural)",
#     "Z7": "Knife-stroke",
#     "Z11": "Three water-strokes (plural II)",
# }
hieroglyph_meanings = {
    "Aa15": "side",
    "Aa26": "cloth",
    "Aa27": "side",
    "Aa28": "building",
    "A55": "death",
    "D1": "head",
    "D2": "face",
    "D10": "protection",
    "D19": "mouth",
    "D21": "tooth",
    "D28": "birth",
    "D34": "sight",
    "D35": "sight",
    "D36": "protection",
    "D39": "strength",
    "D4": "action",
    "D46": "action",
    "D52": "bread",
    "D53": "support",
    "D54": "support",
    "D56": "offering",
    "D58": "back",
    "D60": "sky",
    "D62": "water",
    "D156": "unknown",
    "E1": "divine",
    "E9": "goose",
    "E17": "royalty",
    "E23": "feminine",
    "E34": "power",
    "F4": "feminine",
    "F9": "letter_w",
    "F12": "feminine",
    "F13": "rebirth",
    "F16": "multitude",
    "F18": "speed",
    "F21": "strength",
    "F22": "offering",
    "F23": "strength",
    "F26": "fish",
    "F29": "abundance",
    "F30": "ox",
    "F31": "chaos",
    "F32": "elephant",
    "F34": "ferocity",
    "F35": "tortoise",
    "F40": "sparrow",
    "G1": "motherhood",
    "G4": "lordship",
    "G5": "comb",
    "G7": "offering",
    "G10": "bread",
    "G14": "nobility",
    "G17": "wisdom",
    "G21": "harvest",
    "G25": "nobility",
    "G26": "afterlife",
    "G29": "stability",
    "G35": "boat",
    "G36": "sail",
    "G37": "navigation",
    "G39": "music",
    "G40": "weaving",
    "G43": "protection",
    "G50": "power",
    "H6": "granary",
    "I5": "dominion",
    "I9": "feminine",
    "I10": "royalty",
    "L1": "guardian",
    "M1": "growth",
    "M3": "control",
    "M4": "sail",
    "M8": "wall",
    "M12": "tree",
    "M16": "shrub",
    "M17": "water",
    "M18": "cosmos",
    "M20": "fertility",
    "M23": "liquid",
    "M26": "knowledge",
    "M29": "offering",
    "M40": "bread",
    "M41": "jar",
    "M42": "basket",
    "M44": "sandal",
    "M195": "unknown",
    "N1": "sky",
    "N2": "night",
    "N5": "sun",
    "N14": "star",
    "N16": "land",
    "N17": "field",
    "N18": "island",
    "N19": "duality",
    "N24": "district",
    "N25": "foreign",
    "N26": "mountain",
    "N29": "desert",
    "N30": "mound",
    "N31": "road",
    "N35": "water",
    "N36": "canal",
    "N37": "pool",
    "N41": "well",
    "O1": "house",
    "O4": "granary",
    "O11": "temple",
    "O28": "palace",
    "O29": "fortress",
    "O31": "pyramid",
    "O34": "tomb",
    "O49": "column",
    "O50": "throne",
    "O51": "chair",
    "P1": "boat",
    "P6": "mooring",
    "P8": "drill",
    "P13": "chisel",
    "P98": "weight",
    "Q1": "sandal",
    "Q3": "basket",
    "Q7": "beer",
    "R4": "offering",
    "R8": "authority",
    "S24": "scepter",
    "S28": "life",
    "S29": "stability",
    "S34": "dominion",
    "S42": "truth",
    "T14": "offering",
    "T20": "beer",
    "T21": "wine",
    "T22": "meat",
    "T28": "incense",
    "T30": "ointment",
    "U1": "sun",
    "U7": "plough",
    "U15": "harvest",
    "U28": "star",
    "U33": "pestle",
    "U35": "fire",
    "V4": "road",
    "V6": "stone",
    "V7": "mountain",
    "V13": "cooking",
    "V16": "hill",
    "V22": "desert",
    "V24": "field",
    "V25": "garden",
    "V28": "tree",
    "V30": "purity",
    "V31": "papyrus",
    "W11": "boat",
    "W14": "oar",
    "W15": "sail",
    "W18": "anchor",
    "W19": "fish",
    "W22": "bird",
    "W24": "bee",
    "W25": "scarab",
    "X1": "bread",
    "X6": "beer",
    "X8": "offering",
    "Y1": "cloth",
    "Y2": "flax",
    "Y3": "net",
    "Y5": "basket",
    "Z1": "plural",
    "Z7": "violence",
    "Z11": "plural",
}
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
labels_path = 'models/labels.txt'
model_path = 'models/MobileNetV3_traced_model.pt'

# Load labels
label_map = {}
with open(labels_path, 'r') as f:
    for line in f:
        idx, class_name = line.strip().split(': ')
        label_map[int(idx)] = class_name
print(f"Loaded {len(label_map)} labels from '{labels_path}'")

# Load model
model = torch.jit.load(model_path, map_location=device)
model.to(device)
model.eval()

# Define image transformation
test_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    """Render the home page with the hero and features sections."""
    return render_template('index.html')

@app.route('/lookup', methods=['GET', 'POST'])
def lookup():
    """Handle hieroglyph lookup form submission and display results."""
    if request.method == 'POST':
        code = request.form.get('code', '').strip()
        if not code:
            return render_template('lookup.html', error="Please enter a Gardiner code.")
        meaning = hieroglyph_meanings.get(code, "Meaning not found")
        if meaning == "Meaning not found":
            return render_template('lookup.html', error=f"The Gardiner code '{code}' was not found in our database.")
        return render_template('lookup.html', code=code, meaning=meaning)
    return render_template('lookup.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    """Handle image classification form submission and display results."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('classify.html', error="No file uploaded")
        file = request.files['file']
        if file.filename == '':
            return render_template('classify.html', error="No file selected")
        true_code = request.form.get('true_code', '').strip()
        try:
            # Process the image
            image = Image.open(file.stream)
            grayscale = image.convert('L')
            image_rgb = Image.merge('RGB', (grayscale, grayscale, grayscale))
            image_tensor = test_transform(image_rgb).unsqueeze(0).to(device)

            # Get prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                _, predicted = torch.max(outputs, 1)
                pred_label = predicted.item()

            # Map prediction to Gardiner code and meaning
            gardiner_code = label_map.get(pred_label, "Unknown")
            meaning = hieroglyph_meanings.get(gardiner_code, "Meaning not found")

            # Convert images to base64 for display
            original_buffer = BytesIO()
            image.save(original_buffer, format="PNG")
            original_base64 = base64.b64encode(original_buffer.getvalue()).decode('utf-8')
            grayscale_buffer = BytesIO()
            grayscale.save(grayscale_buffer, format="PNG")
            grayscale_base64 = base64.b64encode(grayscale_buffer.getvalue()).decode('utf-8')

            return render_template('classify.html',
                                   original=original_base64,
                                   grayscale=grayscale_base64,
                                   gardiner_code=gardiner_code,
                                   meaning=meaning,
                                   true_code=true_code if true_code else None)
        except Exception as e:
            return render_template('classify.html', error=str(e))
    return render_template('classify.html')

@app.route('/dictionary')
def dictionary():
    """Render the dictionary page with all hieroglyphs."""
    return render_template('dictionary.html', hieroglyph_meanings=hieroglyph_meanings)

@app.route('/api/lookup/<code>')
def api_lookup(code):
    """API endpoint to return the meaning of a Gardiner code in JSON."""
    meaning = hieroglyph_meanings.get(code, "Meaning not found")
    return jsonify({'gardiner_code': code, 'meaning': meaning})

@app.route('/api/classify', methods=['POST'])
def api_classify():
    """API endpoint to classify an image and return prediction in JSON."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    try:
        # Process the image
        image = Image.open(file.stream)
        grayscale = image.convert('L')
        image_rgb = Image.merge('RGB', (grayscale, grayscale, grayscale))
        image_tensor = test_transform(image_rgb).unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, predicted = torch.max(outputs, 1)
            pred_label = predicted.item()

        # Map prediction to Gardiner code and meaning
        gardiner_code = label_map.get(pred_label, "Unknown")
        meaning = hieroglyph_meanings.get(gardiner_code, "Meaning not found")

        return jsonify({
            'gardiner_code': gardiner_code,
            'meaning': meaning
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7000)