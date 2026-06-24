#!/usr/bin/env python3
"""Surgical differential/pattern-reasoning examples for run-6.

run-5 over-committed to a (wrong) disease on cases that should be a differential:
broiler-dying-at-night -> "Newcastle"; seedlings-cut-at-base+grubs (cutworm) -> "nematodes".
Root cause: 45% "this is disease X" vs 10% differential -> the model learned to commit.
These teach the opposite where it matters: read the PATTERN, give a ranked differential, name
the DISCRIMINATING check, and end with a question — do NOT name one disease confidently when
the pattern is ambiguous. No dose numbers. category=differential_uncertain.
"""
import json


def row(u, a):
    return {"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}],
            "category": "differential_uncertain", "topic": "pattern"}


EX = [
 # ── poultry sudden / night death — DON'T jump to a virus ──────────────────────
 row("My broilers are dying two at a time at night but look fine during the day.",
     "Steady night losses in pairs with healthy daytime birds point to management or environment more than a fast disease. Most likely is piling or smothering — birds crowd into a corner from cold, a draught, or a night fright and the ones underneath suffocate; you'd find them heaped together. Next is sudden death syndrome in fast-growing broilers, found on their backs and scattered, not piled. A night predator is also possible — check for wounds or gaps in the house. In the morning, are the dead birds piled in one corner, or scattered and lying on their backs looking healthy?"),
 row("Several of my chickens died suddenly overnight with no warning signs.",
     "Sudden overnight deaths with no prior signs have a few likely causes, so let's not assume the worst disease yet. Common ones are heat or suffocation if the house was closed and crowded, piling in cold or fright, or poisoning from spoiled feed or a toxin. A peracute disease like fowl cholera or Newcastle is possible but usually shows some sickness first or keeps killing. How many died of how many, were they crowded or hot overnight, and are the survivors completely normal today? That tells us where to look."),
 row("My layers suddenly started dropping dead, a few each day, and the rest seem okay.",
     "A steady trickle of deaths with an otherwise normal-looking flock can mean several things. Heat stress, internal parasites, or a chronic problem like fatty liver in well-fed layers can each cause this. An infectious disease is possible too but usually spreads and worsens. Check the droppings, the combs (pale or normal), the feed and water, and whether it is the heaviest birds dying. What do the dead birds look like inside if you open one, and how are the droppings?"),
 # ── pest damage — discriminate, don't guess the wrong pest ────────────────────
 row("My maize seedlings were cut clean off at the base overnight.",
     "A clean cut at the soil line overnight is the classic sign of cutworms — grey grubs that hide just below the surface and sever stems at night. Don't confuse it with other problems: root-knot nematodes cause knotty galls on the roots and stunting (not a clean cut), while termites tunnel into the stem base in dry conditions. Dig gently around a freshly cut plant in the morning — if you find a curled grey grub, it's cutworm. Are the plants cut cleanly through at the base, and do you find grubs in the topsoil?"),
 row("My maize plants are stunted and yellow in patches, and when I pull them the roots have knotty swellings.",
     "Knotty swellings on the roots with patchy stunting and yellowing point to root-knot nematodes, not a leaf disease or simple hunger. The galls block water and nutrient uptake, so the tops look starved. It's easy to mistake for nitrogen shortage, but nitrogen yellowing is even across the field and the roots look normal. Are the swellings actual lumps on the roots, and is the problem in patches rather than the whole field?"),
 row("Something is eating my cabbage — there are large ragged holes in the leaves.",
     "Large ragged holes usually mean a chewing caterpillar rather than a sap-sucker or a disease. The main suspects are diamondback moth larvae, cabbage loopers, or armyworms — all leaf-chewers you can often find on the undersides. Sucking pests like aphids instead leave curled, sticky leaves, and disease leaves spots, not holes. Turn over a few leaves: do you see green caterpillars and dark droppings, and are the holes chewed rather than spotted?"),
 row("My maize has holes in the whorl and the young leaves look ragged with sawdust-like material.",
     "Ragged young leaves with sawdust-like frass in the whorl is the signature of a caterpillar feeding down in the whorl — fall armyworm or stalk borer are the usual ones. It's different from cutworm (which cuts seedlings at the base) and from disease (which makes spots or streaks). Open the whorl and look: do you see a caterpillar and moist frass inside, and is the damage concentrated in the new central leaves?"),
 # ── crop patterns — patchy vs uniform vs spreading ───────────────────────────
 row("Patches of my bean field are stunted and yellow while the rest looks healthy.",
     "Patchiness is the clue — a problem in patches is usually soil, water, or roots, not something airborne. Most likely is waterlogging or compaction in the low or hard spots starving the roots of air; next is a root problem like nematodes or rot; a localised nutrient shortage is also possible. A spreading disease would usually move out in a front rather than sit in fixed patches. Do the bad patches match where water sits or the soil is hardest, and what do the roots look like there?"),
 row("My whole field of young maize turned pale yellow a few days after heavy rain.",
     "A sudden, field-wide yellowing right after heavy rain points to the rain, not a disease. The leading cause is waterlogging — saturated soil starves roots of air and washes nitrogen below them, so the crop pales all over at once. Nitrogen leaching rides along with it. A disease usually starts in spots and spreads, not everywhere together. Is the water draining away now, and is the yellowing fairly even across the whole field rather than in patches?"),
 row("Brown spots on my tomato leaves are spreading quickly in this wet weather.",
     "Fast-spreading brown spots in wet weather usually mean a fungal or bacterial leaf disease, but which one changes what you do. Early blight makes target-like rings on older leaves and spreads slowly; late blight makes greasy, fast-enlarging blotches, sometimes with white mould underneath, and races through a crop in days; bacterial spot makes small dark specks with yellow halos. Look closely: are the spots large and greasy with pale mould beneath (late blight, urgent), ringed on old leaves (early blight), or tiny with yellow halos (bacterial)?"),
 # ── wilting — vascular vs water vs root ──────────────────────────────────────
 row("My tomato plants wilt in the afternoon but recover overnight, and the soil is moist.",
     "Wilting on moist soil that recovers overnight is a useful clue — the roots or stem aren't moving water fast enough in the heat. The main possibilities are a soil-borne vascular wilt (fusarium or bacterial wilt) starting to block the stem, root or stem rot from soil staying too wet, or root-knot nematodes damaging the roots. Cut a low stem and look inside for browning, and pull a plant to check the roots. Is the inside of the lower stem browned, and do the roots look rotted or knotted?"),
 row("A few of my banana plants are wilting and the leaves are turning yellow from the older ones up.",
     "Yellowing and wilting climbing from the older leaves up, on scattered plants, has two serious possibilities to tell apart. Panama disease (fusarium wilt) blocks the plant's vessels — cut the pseudostem and you'll see brown or reddish streaks inside. Bacterial wilt instead often yellows the whole plant fast and oozes a yellowish slime from a cut stem. A simple nutrient shortage wouldn't wilt the plant or discolour the inside. Cut across a affected pseudostem — do you see brown vascular streaks, or a slimy ooze?"),
 # ── leaf yellowing — nutrient vs disease vs water ────────────────────────────
 row("The lower leaves of my maize are yellowing while the top stays green.",
     "Yellowing that starts on the oldest lower leaves and works upward, with the top staying green, usually means a mobile nutrient is being moved up to the new growth — most often nitrogen, especially on sandy soil or after heavy rain. If instead it were between the veins on the youngest leaves, I'd think of a different nutrient; if there were spots or streaks, a disease or virus. Is the yellowing even across the older leaves and worst at the bottom, with no spots or insects?"),
 row("My citrus leaves are yellowing between the veins while the veins stay green.",
     "Yellowing between the veins with the veins staying green is a fairly specific pattern — a micronutrient shortage, usually iron or zinc, common on high-pH or waterlogged soils where roots can't take them up. Iron shows a fine net on the youngest leaves; zinc gives blotchy mottling and small leaves. A nitrogen shortage instead yellows the whole older leaf evenly. Is it the newest leaves most affected, with the vein lines still green, and is your soil heavy or often wet?"),
 # ── post-harvest / storage pattern ───────────────────────────────────────────
 row("My stored maize is developing mould even though I thought it was dry.",
     "Mould in storage almost always comes back to moisture, but let's find the source. The grain may have gone in above safe dryness, or it picked up damp from a leaking roof, a damp floor, or condensation in a sealed container that warmed and cooled. Insect activity can also raise moisture and heat in a pocket. Check whether the mould is general or in a damp patch, whether the store leaks or sweats, and whether the grain felt truly hard and dry going in. Is the mould all through, or worst near the floor, walls, or a leak?"),
 # ── goat/poultry condition (out-of-domain-aware differential) ────────────────
 row("My goats are losing weight even though they are eating well.",
     "Losing condition on a good appetite is a classic sign, and the top cause to rule out is internal parasites such as worms, which drain the animal while it keeps eating — check the lower eyelid for paleness. Poor-quality feed that fills but doesn't nourish, dental problems in older goats, or a chronic illness can also do it. Since goats are outside what I treat, an animal-health worker should confirm with a dung test, but you can start by checking eyelid colour and recent deworming. Are the eyelids and gums pale, and when were they last dewormed?"),
 row("My hens' eggshells have gone thin and crack easily lately.",
     "Thin, easily-cracked shells usually trace to one of a few things. The commonest is a calcium or vitamin D shortage in the diet — laying hens need a steady calcium source like shell grit or limestone. Heat stress also thins shells, as can ageing hens late in lay, and certain diseases. Look at the whole picture: the hens' age and how long in lay, whether they get a separate calcium source, and how hot the house gets. Are the birds old in their laying cycle, and do they have grit or a shell source available?"),
]


def main():
    with open("diff_set.jsonl", "w", encoding="utf-8") as f:
        for r in EX:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote diff_set.jsonl: {len(EX)} pattern-differential examples")


if __name__ == "__main__":
    main()
