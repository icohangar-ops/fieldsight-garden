#!/usr/bin/env python
"""Writes care/fixes.json: a 3-step, gardener-safe fix per class, with sources.
Content is paraphrased from the cited university extension pages (cultural controls first;
chemicals only as 'a product labeled for this disease on this crop, follow the label').
Source URLs were checked to return HTTP 200 on 2026-10-06 (content not re-verified line by line)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fieldsight.labels import CLASSES  # noqa: E402

UMN = "https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota"
UMN_VEG = "https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management"
SOURCES = {
    "umn_apple_scab": ("Apple scab", "University of Minnesota Extension", f"{UMN}/yard-and-garden-problems/apple-scab"),
    "umn_cedar_apple_rust": ("Cedar-apple rust", "University of Minnesota Extension", f"{UMN}/yard-and-garden-problems/cedar-apple-rust"),
    "umn_bacterial_spot": ("Bacterial spot of tomato and pepper", "University of Minnesota Extension", f"{UMN_VEG}/bacterial-spot-of-tomato-and-pepper"),
    "umn_gray_leaf_spot": ("Gray leaf spot on corn", "University of Minnesota Extension", "https://extension.umn.edu/agriculture/crop-production/corn/gray-leaf-spot-on-corn"),
    "umn_nclb": ("Northern corn leaf blight", "University of Minnesota Extension", "https://extension.umn.edu/agriculture/crop-production/corn/northern-corn-leaf-blight"),
    "umn_common_rust": ("Common rust on corn", "University of Minnesota Extension", "https://extension.umn.edu/agriculture/crop-production/corn/common-rust-on-corn"),
    "osu_black_rot": ("Grape black rot (PLPATH-FRU-24)", "Ohio State University Extension (Ohioline)", "https://ohioline.osu.edu/factsheet/plpath-fru-24"),
    "umass_black_rot": ("Grape IPM: Black rot", "UMass Amherst Extension", "https://www.umass.edu/agriculture-food-environment/fruit/fact-sheets/grape-ipm-black-rot"),
    "umn_early_blight": ("Early blight in tomato and potato", "University of Minnesota Extension", f"{UMN_VEG}/early-blight-in-tomato-and-potato"),
    "umn_late_blight": ("Late blight", "University of Minnesota Extension", f"{UMN_VEG}/late-blight"),
    "uw_late_blight": ("Late blight", "University of Wisconsin-Madison Extension", "https://hort.extension.wisc.edu/articles/late-blight/"),
    "umn_cucurbit_pm": ("Powdery mildew of cucurbits", "University of Minnesota Extension", f"{UMN_VEG}/powdery-mildew-of-cucurbits"),
    "ucipm_pm_veg": ("Powdery mildew on vegetables", "UC Statewide IPM Program", "https://ipm.ucanr.edu/home-and-landscape/powdery-mildew-on-vegetables/"),
    "umn_tomato_leaf_spots": ("Tomato leaf spot diseases", "University of Minnesota Extension", f"{UMN}/yard-and-garden-problems/tomato-leaf-spot-diseases"),
    "uw_septoria": ("Tomato Septoria leaf spot", "UW-Madison Vegetable Pathology", "https://vegpath.plantpath.wisc.edu/diseases/tomato-septoria-leaf-spot/"),
    "umn_leaf_mold": ("Tomato leaf mold", "University of Minnesota Extension", f"{UMN_VEG}/tomato-leaf-mold"),
    "umn_tomato_viruses": ("Tomato viruses", "University of Minnesota Extension", f"{UMN_VEG}/tomato-viruses"),
    "ucipm_tylcv": ("Tomato yellow leaf curl", "UC Statewide IPM Program", "https://ipm.ucanr.edu/agriculture/tomato/tomato-yellow-leaf-curl/"),
    "umn_grow_tomatoes": ("Growing tomatoes", "University of Minnesota Extension", f"{UMN}/growing-tomatoes"),
    "umn_grow_apples": ("Growing apples", "University of Minnesota Extension", f"{UMN}/growing-apples"),
    "umn_grow_blueberries": ("Growing blueberries in the home garden", "University of Minnesota Extension", f"{UMN}/growing-blueberries-in-the-home-garden"),
    "umn_grow_raspberries": ("Growing raspberries in the home garden", "University of Minnesota Extension", f"{UMN}/growing-raspberries-in-the-home-garden"),
    "umn_grow_strawberries": ("Growing strawberries in the home garden", "University of Minnesota Extension", f"{UMN}/growing-strawberries-in-the-home-garden"),
    "umn_grow_grapes": ("Growing grapes in the home garden", "University of Minnesota Extension", f"{UMN}/growing-grapes-in-the-home-garden"),
    "umn_grow_peppers": ("Growing peppers", "University of Minnesota Extension", f"{UMN}/growing-peppers"),
    "umn_grow_stone_fruit": ("Growing stone fruits in the home garden", "University of Minnesota Extension", f"{UMN}/growing-stone-fruits-in-the-home-garden"),
    "umn_soybean": ("Soybean production", "University of Minnesota Extension", "https://extension.umn.edu/agriculture/crop-production/soybean"),
    "cpn_encyclopedia": ("Field crop disease encyclopedia", "Crop Protection Network (land-grant universities)", "https://cropprotectionnetwork.org/encyclopedia"),
}

FUNGICIDE = "If it keeps spreading, use a fungicide labeled for {d} on {c} and follow the label exactly."

FIXES = {
    "apple scab": ("fungal", [
        "Rake up and remove fallen apple leaves this autumn; the scab fungus overwinters in them.",
        "Prune for an open canopy so leaves dry fast after rain.",
        "For a chronic problem, plant scab-resistant varieties, or start a labeled fungicide at bud break next spring."],
        ["umn_apple_scab"]),
    "apple rust": ("fungal", [
        "Pick off badly spotted leaves; rust needs nearby junipers or red cedars to complete its cycle.",
        "Cut off orange or brown galls on nearby junipers in late winter, before they swell in spring rain.",
        "For new plantings, choose rust-resistant apple varieties; labeled fungicides only protect new leaves in spring."],
        ["umn_cedar_apple_rust"]),
    "apple healthy": ("healthy", [
        "Looks healthy. Water deeply at the base during dry spells.",
        "Rake up fallen leaves and fruit in autumn to cut next year's disease.",
        "Scout weekly for spots, curling or pests, especially after wet weather."],
        ["umn_grow_apples"]),
    "bell pepper leaf spot": ("bacterial", [
        "Remove spotted leaves and avoid touching plants while they are wet; bacteria spread by splash and hands.",
        "Water at the soil, not the leaves, and mulch to stop soil splashing up.",
        "Next year, rotate peppers and tomatoes to a new bed and use clean seed or transplants."],
        ["umn_bacterial_spot"]),
    "bell pepper healthy": ("healthy", [
        "Looks healthy. Keep soil evenly moist and water at the base.",
        "Mulch around plants to hold moisture and keep soil off the leaves.",
        "Check leaf undersides weekly for spots or aphids."],
        ["umn_grow_peppers"]),
    "blueberry healthy": ("healthy", [
        "Looks healthy. Keep soil moist and acidic, and mulch with wood chips or pine needles.",
        "Prune out old, weak or dead canes in late winter.",
        "Watch for spotted or reddening leaves and shriveled berries."],
        ["umn_grow_blueberries"]),
    "cherry healthy": ("healthy", [
        "Looks healthy. Water deeply during dry spells and keep mulch off the trunk.",
        "Prune in late winter for airflow and remove dead or cracked wood.",
        "Rake up fallen leaves in autumn to reduce leaf-spot disease next year."],
        ["umn_grow_stone_fruit"]),
    "corn gray leaf spot": ("fungal", [
        "Note the long, narrow, gray-tan lesions between the leaf veins; it spreads from infected corn residue.",
        "After harvest, chop and bury or remove corn residue, and rotate corn to a different spot next year.",
        "Choose resistant hybrids next season; fungicides are rarely worthwhile in a home garden."],
        ["umn_gray_leaf_spot", "cpn_encyclopedia"]),
    "corn leaf blight": ("fungal", [
        "Long cigar-shaped tan lesions usually mean northern corn leaf blight, which spreads from old corn residue.",
        "Remove or bury corn debris after harvest and rotate corn elsewhere next year.",
        "Plant resistant hybrids next season; in a small garden the crop usually still yields."],
        ["umn_nclb", "cpn_encyclopedia"]),
    "corn rust": ("fungal", [
        "Small, powdery, rust-colored pustules on both leaf sides are common rust; spores blow in on the wind.",
        "Late-season rust rarely hurts sweet corn yield, so just keep the plants watered and fed.",
        "Choose rust-resistant varieties next year if it shows up early and heavily."],
        ["umn_common_rust", "cpn_encyclopedia"]),
    "grape black rot": ("fungal", [
        "Remove spotted leaves and any shriveled, black 'mummy' berries now, and bag them; don't compost them.",
        "In winter, prune out infected canes and tendrils and rake up mummies under the vine.",
        "Keep vines sunny and open; for repeat problems, use a fungicide labeled for black rot from before bloom."],
        ["osu_black_rot", "umass_black_rot"]),
    "grape healthy": ("healthy", [
        "Looks healthy. Train vines for sun and airflow.",
        "Prune hard in late winter and remove old clusters and debris.",
        "Scout weekly from bloom for spots on leaves and berries."],
        ["umn_grow_grapes"]),
    "peach healthy": ("healthy", [
        "Looks healthy. Water deeply during dry spells and mulch, keeping it off the trunk.",
        "Prune to an open center in late winter for light and airflow.",
        "Watch spring leaves for red, puckered curl and remove rotting fruit promptly."],
        ["umn_grow_stone_fruit"]),
    "potato early blight": ("fungal", [
        "Remove the lower leaves with target-like brown rings and keep plants well fed; stressed plants get it worse.",
        "Water at the soil in the morning and mulch to stop splash.",
        "Rotate potatoes and tomatoes out of this bed for at least two years. " + FUNGICIDE.format(d="early blight", c="potatoes")],
        ["umn_early_blight"]),
    "potato late blight": ("oomycete", [
        "Act today: late blight spreads fast. Pull infected plants, bag them and put them in the trash, not the compost.",
        "Check nearby potatoes and tomatoes daily for greasy dark blotches with white fuzz underneath.",
        "Next year, plant certified disease-free seed potatoes and destroy any volunteer potatoes."],
        ["umn_late_blight", "uw_late_blight"]),
    "raspberry healthy": ("healthy", [
        "Looks healthy. Keep rows narrow and thinned for airflow.",
        "After harvest, cut fruited floricanes to the ground.",
        "Water at the base and watch for spotted canes or wilting tips."],
        ["umn_grow_raspberries"]),
    "soybean healthy": ("healthy", [
        "Looks healthy. Scout every week or two, checking leaves top and bottom.",
        "Rotate soybeans with a non-legume crop to reduce soil-borne disease.",
        "If spots appear, compare them against an extension disease guide before treating."],
        ["umn_soybean", "cpn_encyclopedia"]),
    "squash powdery mildew": ("fungal", [
        "Remove the worst white-dusted leaves, but keep enough foliage to shade the fruit.",
        "Give plants sun and space, and water at the base.",
        "Plant mildew-resistant varieties next year. " + FUNGICIDE.format(d="powdery mildew", c="squash")],
        ["umn_cucurbit_pm", "ucipm_pm_veg"]),
    "strawberry healthy": ("healthy", [
        "Looks healthy. Water at the base in the morning so leaves dry quickly.",
        "Mulch with straw to keep berries off the soil.",
        "Thin runners and remove old leaves after harvest for airflow."],
        ["umn_grow_strawberries"]),
    "tomato bacterial spot": ("bacterial", [
        "Pick off spotted leaves when plants are dry and wash your hands after; bacteria spread by splash and touch.",
        "Water at the soil, mulch, and stake plants so leaves dry fast.",
        "Rotate tomatoes and peppers to a new bed for 2 to 3 years, and don't save seed from these plants."],
        ["umn_bacterial_spot", "umn_tomato_leaf_spots"]),
    "tomato early blight": ("fungal", [
        "Pinch off lower leaves with brown target-ring spots, up to a third of the plant.",
        "Mulch the soil and water at the base in the morning; stake for airflow.",
        "Rotate tomatoes out of this bed next year. " + FUNGICIDE.format(d="early blight", c="tomatoes")],
        ["umn_early_blight", "umn_tomato_leaf_spots"]),
    "tomato healthy": ("healthy", [
        "Looks healthy. Water deeply at the base and keep leaves dry.",
        "Mulch the soil and stake or cage the plant for airflow.",
        "Check lower leaves weekly for spots and pinch off any you find early."],
        ["umn_grow_tomatoes", "umn_tomato_leaf_spots"]),
    "tomato late blight": ("oomycete", [
        "Act today: late blight spreads fast. Pull infected plants, bag them and put them in the trash, not the compost.",
        "Check nearby tomatoes and potatoes daily for greasy dark blotches and white fuzz under the leaves.",
        "Next year, choose late-blight-resistant varieties and remove volunteer tomatoes and potatoes."],
        ["umn_late_blight", "uw_late_blight"]),
    "tomato leaf mold": ("fungal", [
        "Remove leaves with yellow tops and olive-brown fuzz underneath.",
        "Cut humidity: space and prune plants, and ventilate greenhouses or high tunnels.",
        "Water at the soil, and choose leaf-mold-resistant varieties next time."],
        ["umn_leaf_mold"]),
    "tomato mosaic virus": ("viral", [
        "There is no cure: remove and bag the mottled plant so it can't spread to others.",
        "Wash hands and disinfect tools and stakes; the virus spreads by touch.",
        "Plant resistant varieties and clean seed next season."],
        ["umn_tomato_viruses"]),
    "tomato septoria leaf spot": ("fungal", [
        "Pinch off lower leaves with many small, dark-edged spots, up to a third of the plant.",
        "Water at the base, mulch, and stake to keep leaves dry; remove plant debris at season end.",
        "Rotate tomatoes for 1 to 2 years. " + FUNGICIDE.format(d="Septoria leaf spot", c="tomatoes")],
        ["umn_tomato_leaf_spots", "uw_septoria"]),
    "tomato yellow leaf curl virus": ("viral", [
        "There is no cure: remove and bag curled, yellowed plants, especially young ones.",
        "Control whiteflies, which spread this virus; check leaf undersides and use reflective mulch or row covers.",
        "Plant resistant varieties and avoid planting next to infected crops."],
        ["ucipm_tylcv"]),
}
assert set(FIXES) == set(CLASSES), set(CLASSES) ^ set(FIXES)

out = {
    "version": 1,
    "verified": "Source URLs returned HTTP 200 on 2026-10-06; wording is a short paraphrase of each page.",
    "disclaimer": ("Guidance is general and for home gardeners. Diagnosis from one photo can be wrong; "
                   "confirm with your local extension office before removing plants or spraying. Always "
                   "read and follow pesticide labels."),
    "sources": {k: {"title": t, "publisher": p, "url": u} for k, (t, p, u) in SOURCES.items()},
    "fixes": {c: {"type": FIXES[c][0], "steps": FIXES[c][1], "sources": FIXES[c][2]} for c in CLASSES},
}
for c, f in out["fixes"].items():
    assert len(f["steps"]) == 3, c
    assert all(s in SOURCES for s in f["sources"]), c
(ROOT / "care" / "fixes.json").write_text(json.dumps(out, indent=2))
print(f"wrote care/fixes.json with {len(out['fixes'])} classes, {len(SOURCES)} sources")
