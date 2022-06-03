This document delineates the format of the output data for the SitEnt Stativity Classifier. The directory "sd_contrast_output" contains sample output.

Clause: transformed clause belonging to contrast set

origClause: the original clause as pulled directly from the original test set

mainVerb: the main verb of the sentence automatically identified for the original clause, pulled from original test set

mainReferent: the main referent automatically identified for the original clause, pulled from original test set

contrastVerb: the main verb of the contrast clause, manually inserted by contrast set annotator

Contrast: predicted label corresponding to contrast clause (if only first clause was annotated, "Contrast" for non-initial clauses corresponds to the the gold standard label from the original test set)

Strategy: transformation strategy for transforming original test set clause into contrast set clause such that the label is reversed (STATIVE -> DYNAMIC, DYNAMIC -> STATIVE). Manually inserted by contrast set annotator, categories are as follows:

DYNAMIC -> STATIVE:
1. THOUGHT VERB - Demote a dynamic verb from main to secondary verb by moving it into the subordinate THEME role for verbs of thinking, believing, or feeling.
2. COPULA - Replace the main verb with a simple predication headed by an English copular verb.
3. DESCRIPTIVE VERB - Replace the dynamic action with a descriptive verb, effectively reconfiguring the dynamic action as stative properties of the subject noun.
4. LIGHT VERB - Use the possessive light verb construction with "have" to make "have" the new main verb.
5. SEMI-MODAL - Use a semi-modal verb (e.g. need to, ought to) marking deontic modality, which concerns the speaker's requirements and desires, as a "thought" or "emotion" from the speaker, who can be an unspecified authority with no referent.
6. DOWNGRADE TO PPL - Remove the main verb from the clause by transforming it into a perfect passive participle that lends itself more to a descriptive, adjectival reading than a verbal reading.
7. ORDER - Switch the order of the clauses and insert a descriptive verb as the new main verb.

STATIVE -> DYNAMIC:
1. NEW PARTICIPANT - Choose a synonymous verb that introduces an agent who participates in a dynamic synonym of the original verb.
2. INSERT VERB - Replace a stative verb (typically the copula) with a dynamic verb or insert a dynamic verb as the new main verb, of which the original stative verb is a dependent. 
3. BECOMING - Replace standard copula with an inflected form of the verbs "to become" or "to get" and their synonyms. This preserves the copula's predicating structure while reformulating the event as dynamic.
4. UPGRADE - Upgrade a perfect passive participle or subordinate STATIVE verb to the main verb of the clause by adding a helping verb or deleting the main STATIVE verb.
5. HEAVY VERB - Replace a light verb construction like "have" with a heavy, dynamic verb.