import os
import whisper
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, set_seed
import torch
from huggingface_hub import login
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
import werpy
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path # for finding files
from dotenv import load_dotenv # for storing the huggingface token 
np.random.seed(42)# set seed for whole env
torch.manual_seed(42)
print("packages imported")

# finding the directory 
try:
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
except NameError:
    # Fallback for REPLs
    script_dir = Path.cwd()

# KEY PATHS
output_folder = script_dir / "outputs"
audio_folder = script_dir / "audio"
output_folder.mkdir(exist_ok=True)

# Find and Load Audio File
try:
    # Find the first .mp3 file in the audio folder
    sg_01_path = next(audio_folder.glob("*.wav"))
    sg_01 = str(sg_01_path)
    print(f"   Audio file found: {sg_01_path.name}")
except StopIteration:
    print(f"   ERROR: No audio file found in {audio_folder}")
    exit()

# -------- Load the base whisper_OpenAI model
print("Loading Whisper model...")
model = whisper.load_model("large-v3-turbo") # small, large, large-v3-turbo
print("base_whisper model loaded.")

# Transcribe the audio file 
print("\nStarting transcription...")
large_whisper_result = model.transcribe(
  sg_01)
print("\nBase_model transcription finished")
print(large_whisper_result['text'])

# ----------- Swiss German Fine tuned model 
print("loading fine-tuned Swiss German model from huggingface.")
# Set the device for optimal performance (MPS for Apple Silicon, CUDA for Nvidia)
device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Create the transcription pipeline using the Hugging Face model identifier
# Your token from https://huggingface.co/settings/tokens
# either create a .env file with "HF_TOKEN="YOUR TOKEN"" or alternativley use the following liine and comment the if statement out. 

# Use:
HF_TOKEN = "YOUR TOKEN"

# and Ccmment out the follwoing if using HF_TOKEN="YOUR TOKEN" 
#load_dotenv()
#HF_TOKEN = os.getenv("HF_TOKEN")

#if not HF_TOKEN:
#    raise ValueError("Hugging Face token not found! Please create a .env file.")
# End Comment out 

login(token=HF_TOKEN)
print("Successfully logged into hugginface.")
# need token to accesses gated communities within huggingface access token permissions

# finetuned sg model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "Flurin17/whisper-large-v3-turbo-swiss-german"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

set_seed(42) # set seed
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device
)

# Transcribe a Swiss German audio file
swiss_german_result = pipe(
  sg_01, 
  return_timestamps=True, language = 'de'
)
print("fine tuned model transcription finished")
print(swiss_german_result["text"])

# summary of data 
reference_transcription = "SRF Audio. Regionaljournal Zürich-Schaffhausen. Bäume umgefallen und Flugzeuge durchgestartet. Die Region hat einen stürmischen Donnerstag erlebt. Und zuschauen beim Kaiserschnitt. Das ist in Spitäler in der Region ein Bedürfnis. Am Mikrofon Christoph Brunner. Seit gestern Nachmittag ist der Sturm Benjamin über die Schweiz gefegt. In der Region Zürich-Schaffhausen hat der Äste abgerissen und Bauabschrankungen umeinander gewirbelt. Im Schaffhausischen Buttenharz ist ein Baum umgeknickt, wie BRK News berichtete. Grössere Schäden sind nicht bekannt. Wegen dem Wind konnte gestern in Adlischwil die Felseneck-Bahn nicht fahren und am Flughafen Zürich hat es teilweise ziemlich wackelnde Landemanöver gegeben. Eine Flughafen-Sprecherin sagt gegenüber SRF: „Wegen dem Sturm wurden gestern 15 An-und Abflüge annulliert .” Antisemitismus-kritik am Stars in Town Festival. Nächstes Jahr tritt der US-Musiker Macklemore am Schaffhauser Festival auf. Er ist in letzter Zeit mit scharfer Israel-Kritik aufgefallen. Im Musikvideo hat er die Situation im kriegsversehrten Gaza mit dem Holocaust verglichen, das ist als antisemitistisch einzustufen. Das sagt der Generalsekretär vom Schweizerisch-Israelitischen Gemeindebund, der Jonathan Kreutner, in den Schaffhauser-Nachrichten. Auch der SB-Kantonsrat Patrick Portmann kritisiert, der Macklemore verbreitet  Verschwörungstheorien über jüdische Weltherrschaft. Die Verantwortlichen vom Festival verteidigen ihre Entscheidung, Kritik an der israelischen Regierung, seie nicht automatisch Antisemitismus. Kritik an der Stadt Zürich wegen einer Baumfallaktion. Es geht um eine Allee von Buchen am Ütliberg. Ein Teil von den Bäumen ist bis zu 100 Jahre alt. Die Grünen haben über 2000 Unterschriften gesammelt, dass die Stadt die Buchen nicht fällt. Am Mittwoch Nachtmittag haben sie die Petition an die zuständige Stadträtin übergegeben und haben dort erfahren, dass ein Teil der Buchen schon Ende letzte Woche gefällt wurden nicht, wie die Hamedia-Zeitungen schreiben. Die Grünen kritisieren die Stadt und reden von Geringschätzig. Die Zahl der Kaiserschnittgeburten steigt. In der Region Zürich-Schaffhausen sind letztes Jahr schon zwei von fünf Babys so auf die Welt gekommen. Die Spitäler reagieren darauf und ermöglichen Eltern jetzt, dass sie beim Kaiserschnitt zuschauen können. Peter Schürmann. Der Kaiserschnitt ist ein chirurgischer Eingriff bei dem die Eltern den Moment, bei dem das Kind auf Welt kommt, verpassen weil die Operation mit einem Tuch abdeckt wird. Bis bis Ärzte in London neue Methoden entwickelt haben, die jetzt auch in Winterthur zum Einsatz kommen. Der Bauch von der Frau, die das Kind bekommt, wird zwar weiterhin mit einem Tuch abgedeckt, aber sobald die Ärztin das Baby aus dem Bauch heraushebt, wird eine Art Fenster im Vorhang aufgemacht und die Mutter kann zuschauen, erklärt Leela Sultern-Bayer. Die Chefärztin an der Klinik für Geburtshilfe am Kantonspital Wintertur. Man weiß jetzt aus verschiedenen Untersuchungen, dass die meisten Frauen insgesamt zufrieden sind mit einem Kaiserschnitt, aber doch einige Frauen immer wieder von einer Lücke berichten. Seit dem Sommer können Mütter am KSW dank dem Fenster die Lücke schließe und das stärkt Bindung zum Kind. So die Leela Sultan-Bayer helfe auch gegen postnatale Depressionen. Aber das KSW ist nicht allein mit dem Angebot, auch im Staatsspital Zürich ist das möglich seit über zwei Jahren. Und da macht das mittlerweile jede vierte Mutter, sagt Natalia Conde, die leitende Ärztin für der Frauenklinik. Viele Frauen, die das gseh würdet , das wäre wahnsinnig toll gewesen, das zu sehen und auch zu sehen, wie der Mann dann die Nabelschnur durchtrennt. Wir haben uns gedacht, dass wir das natürlicher machen, obwohl mir bewusst ist, dass das immer noch eine Operation ist. Und genau das, dass es immer noch eine Operation sei, würde viele Frauen davon abhalte durchs Fenster bei der Geburt zusehen, weil man Angst hat, man würde etwas sehen was man gar nicht will, dass sie Blut sehen würden oder sogar ihren offenen Bauch. Ich merke das auch in den Reaktionen, wenn wir Infoabende haben. Wir machen immer den Infoabend bei uns und da erwähnen wir auch, dass wir das Angebot haben. Und dann schauen mich die Frauen manchmal mit grossen Augen an und dann muss man immer sagen: „Nein, man sieht wirklich nur, wie das Baby rauskommt und man sieht keinen Operationssitus, wie man sagt. Also man sieht wirklich, wie ihres  Baby auf die Welt kommt und die, die sich dann für das entscheiden, haben eine sehr schöne Erfahrung, aber es ist halt sehr individuell. Und doch ein Trend. Am Universitätsspital Zürich ist es schon seit 2023 möglich und da machen das schon neun von zehn Frauen. Und in der Klinik Hirschland heißt auf Nachfrage: „Wir bieten das noch nicht an, haben aber die besondrigen Tücher, das möglich machen, schon gekauft, weil die Nachfrage einfach so groß sei. Peter Schummen. Es bleibt euch die Wetterprognose von SRF Meteo. Heute stürmt es nicht mehr, aber es sei noch windig und am Morgen könnte es noch regnen. Dann wäre es aber ziemlich sonnig mit bis zu zwölf Grad. Das war ein Podcast von SRF, produziert im Auftrag der SRG."
print("reference transcription loaded")
print(reference_transcription)
# base whisper model : large_whisper_result["text"],
# SG fine tuned: swiss_german_result["text"],
# reference transcription : reference_transcription

# --------------- Analysises 
print("starting analysis")
# WER
# normalize the texts individually 
whisper_data = werpy.normalize(large_whisper_result["text"])
sg_data = werpy.normalize(swiss_german_result["text"])
reference_data = werpy.normalize(reference_transcription)

# word error rate: (ref, hyp)
# summaries detailing: insertions, deletions, substitutions 
wer_model_ref = werpy.wer([reference_data], [whisper_data])
summary_ref_whis = werpy.summary([reference_data], [whisper_data])

wer_sg_ref = werpy.wer([reference_data], [sg_data])
summary_ref_sg = werpy.summary([reference_data], [sg_data])

wer_model_sg = werpy.wer([sg_data], [whisper_data])
summary_sg_whis = werpy.summary([sg_data], [whisper_data])
print("wer calculated")

# word counts 
print(f"Word count, base model transcription: {len(large_whisper_result["text"].split())}")
print(f"Word count, fine tuned SG model transcription: {len(swiss_german_result["text"].split())}")
print(f"Word count, reference transcription: {len(reference_transcription.split())}") 
# prop table: wer, bleu, ld, deletions, instertions, substitutions
# ref vs whisper
wer1 = summary_ref_whis["wer"]
ld1 = summary_ref_whis["ld"] # Levenstein Distance :This is the total number of edits (substitutions, deletions, and insertions) required to change your hypothesis text into the reference text
ins1 = summary_ref_whis["insertions"]
del1 = summary_ref_whis["deletions"]
subs1 = summary_ref_whis["substitutions"]

# ref vs sg_model
wer2 = summary_ref_sg["wer"]
ld2 = summary_ref_sg["ld"]
ins2 = summary_ref_sg["insertions"]
del2 = summary_ref_sg["deletions"]
subs2 = summary_ref_sg["substitutions"]


# sg_model vs base_whisper
wer3 = summary_sg_whis["wer"]
ld3 = summary_sg_whis["ld"]
ins3 = summary_sg_whis["insertions"]
del3 = summary_sg_whis["deletions"]
subs3 = summary_sg_whis["substitutions"]

# creating prop
data = {
    "wer": [wer1, wer2, wer3],
    "ld": [ld1, ld2, ld3],
    "deletions": [del1, del2, del3],
    "instertions": [ins1, ins2, ins3],
    "substitutions": [subs1, subs2, subs3]
}

cols_labels = ["wer", "ld", "deletions", "instertions", "substitutions"]
index_labels = [
    "reference vs. base_whisper",
    "reference vs. SG_model",
    "SG_model vs. base_whisper"
]

results_table = pd.DataFrame(data, index=index_labels, columns = cols_labels)
# print(results_table)

# saving a results table: html good for word.docx
html_filename = 'model_performance_table.html'
results_table.to_html(html_filename)
print(f"Table successfully saved as: {html_filename}")

#------- data viz. 

# ---- barchart for bleu and wer
wer_to_plot = results_table['wer'].apply(lambda series_inside: series_inside.iloc[0])
df_to_plot = pd.DataFrame({'wer': wer_to_plot,})

# plotting
fig, ax = plt.subplots(figsize=(12, 8))
df_to_plot.plot(
    kind='bar',
    ax=ax,
    legend=True,
    width=0.8,
    colormap='plasma'
)
ax.set_title('WER score by Model Comparisions', fontsize=14)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.0)
plt.xticks(rotation=30, ha = 'center') # Adjusts the labels
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(script_dir / "outputs" / "wer_comparison.png")
print(f"Plot 1 successfully saved")

# ------- barcharts for LD, dels., insers., subs. 
# Use .apply() to clean each column
ld_to_plot = results_table['ld'].apply(lambda s: s.iloc[0])
del_to_plot = results_table['deletions'].apply(lambda s: s.iloc[0])
ins_to_plot = results_table['instertions'].apply(lambda s: s.iloc[0])
subs_to_plot = results_table['substitutions'].apply(lambda s: s.iloc[0])

# Combine them into a new DataFrame
df_error_counts = pd.DataFrame({
    'Levenshtein Dist.': ld_to_plot,
    'Deletions': del_to_plot,
    'Insertions': ins_to_plot,
    'Substitutions': subs_to_plot
})

# plotting
fig, ax = plt.subplots(figsize=(10, 7))
df_error_counts.plot(
    kind='bar',
    ax=ax,
    legend=True,
    colormap='plasma' 
)
ax.set_title('Error Counts', fontsize=16)
ax.set_ylabel('Total Count')
plt.xticks(rotation=30, ha='center') 
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(script_dir / "outputs" / "error_counts_chart.png")
print(f"Plot 2 successfully saved")