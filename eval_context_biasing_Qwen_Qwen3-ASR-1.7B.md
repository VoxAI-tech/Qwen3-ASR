# Context Biasing Evaluation: Qwen/Qwen3-ASR-1.7B

**Date:** 2026-02-06 14:53 UTC
**Dataset:** VoxAI/bk-pl-dataforce-phase1 (mic channel, speech only)
**Samples:** 440
**Audio:** 37.9 min

## Results

| Condition | WER | CER |
|---|---|---|
| No context | 50.62% | 33.49% |
| With menu context | 135.48% | 143.60% |

## Delta

- **WER:** +84.86% (167.6% relative regression)
- **CER:** +110.11% (328.8% relative regression)

## Context String

Menu items provided as system-prompt context:

```
whopper double whopper whopper junior plant whopper plant whopper junior cheeseburger double cheeseburger plant cheeseburger hamburger big king big king XXL bacon king bacon king junior bacon cheese royal bacon cheese whopper chili cheese burger chicken burger crispy chicken chicken royal king whiskey bao king cheester ranch burger summer crunch king smart nuggetsy nuggets plant nuggets chili cheese nuggets frytki duże frytki crispy cebula pepsi pepsi zero cola mirinda 7 up sprite ice tea lipton cappuccino shake lody deser wrap zestaw zestaw dziecięcy chicken lovers
```

## Sample Comparisons

**Utterance 0**
- REF: `eee dzień dobry czy u pana można tutaj te kody podawać tak tak tak kupony yyy bo tutaj mi wyskak wys`
- No context: `dzień dobry czy u pana można tutaj tak kody podawać tak tak tak kupony bo tutaj mi wyskoczył właśnie`
- With context: `dzień dobry czy u pana można tutaj tak kody podawać tak tak tak kupony bo tutaj mi wyskoczył właśnie`

**Utterance 110**
- REF: `double ale bez cebuli bo wy tam dajecie surową cebulę`
- No context: `dabel ale bez cebuli bo wy tam dajecie surową cebulę`
- With context: `double ale bez cebuli bo wy tam dajecie surową cebulę`

**Utterance 220**
- REF: `dzień dobry poproszę tak yyy tego deala tylko że z tym yyy wege yyy cheeseburgerem i do tego frytki`
- No context: `dzień dobry powrócę tak tego dyla tylko że z tym beke czy z burgiem i do tego frittii`
- With context: `dzień dobry powrócę tak tego dila tylko że z tym vegan cheeseburggerem i do tego fritii`

**Utterance 330**
- REF: `dobra`
- No context: `no`
- With context: `staną`

**Utterance 439**
- REF: `dziękuję`
- No context: `ha`
- With context: `nie`
