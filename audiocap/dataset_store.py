from data_laion import DataLaion
import transformers
import typer


app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
def main(dataset_name: str = typer.Option("cahya/audiosnippets-tiny", help="Dataset Name"),
         train_split: float = typer.Option(0.95,  help="Dataset train split"),
         max_rows: int = typer.Option(0,  help="Dataset maximum train rows"),
         dataset_column_audio: str = typer.Option("mp3", help="Dataset Column Audio <mp3|audio.mp3>"),
         dataset_column_metadata: str = typer.Option("json", help="Dataset Column Metadata <json|metadata.json>"),
         dataset_column_file_name: str = typer.Option("sample_id", help="Dataset Column file name <sample_id|segment_filename>"),    
         dataset_column_duration: str = typer.Option("duration", help="Dataset Column Duration <duration|duration_ms>"),
         dataset_column_duration_scale: float = typer.Option(1.0,  help="Dataset Column Duration Scale"),
         dataset_local_dir: str = typer.Option(None, help="Dataset local directory"),
         dataset_with_emotion: bool = typer.Option(False, help="Dataset with_emotion"),
         dataset_with_caption: bool = typer.Option(False, help="Dataset with_caption"),
         dataset_with_detailed_caption: bool = typer.Option(False, help="Dataset with_detailed_caption"),
         dataset_with_transcription: bool = typer.Option(False, help="Dataset with_transcription")) -> None:
    
    processor = transformers.WhisperProcessor.from_pretrained("openai/whisper-base", language="en", task="transcribe")
    data_laion = DataLaion(dataset_name, processor, train_split=train_split, max_rows=max_rows, dataset_column_audio=dataset_column_audio,
        dataset_column_metadata=dataset_column_metadata, dataset_column_file_name=dataset_column_file_name, 
        dataset_column_duration=dataset_column_duration, dataset_column_duration_scale=dataset_column_duration_scale,
        with_emotion=dataset_with_emotion, with_caption=dataset_with_caption, 
        with_detailed_caption=dataset_with_detailed_caption, with_transcription=dataset_with_transcription)
    dataset = data_laion.get_dataset()
    print("Dataset", dataset)
    if dataset_local_dir is None:
        dataset_local_dir = dataset_name.replace("/", "_")
    dataset.save_to_disk(dataset_local_dir)
    print("Done")


if __name__ == "__main__":
    print("Dataset Store")
    app()
    print("Done")