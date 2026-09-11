"""Download IH datasets for veRL training."""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download


def materialize(snapshot, output_dir):
    """Save embedded images and write parquets with local image paths."""
    snapshot, output_dir = Path(snapshot), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "test"]:
        files = sorted((snapshot / "data").glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No {split} parquet shards in {snapshot / 'data'}")
        image_dir = output_dir / "images" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        writer = None
        count = 0
        try:
            for path in files:
                for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                    rows = batch.to_pylist()
                    for row in rows:
                        local_images = []
                        for record in row["images"]:
                            contents = record["bytes"]
                            if contents is None:
                                raise ValueError("Expected self-contained PNG bytes in Hub dataset")
                            target = image_dir / Path(record["path"]).name
                            target.write_bytes(contents)
                            local_images.append({"image": str(target)})
                        row["images"] = local_images
                    schema = pa.schema([
                        pa.field("images", pa.list_(pa.struct([("image", pa.string())])))
                        if field.name == "images" else field
                        for field in batch.schema
                    ])
                    table = pa.Table.from_pylist(rows, schema=schema)
                    if writer is None:
                        writer = pq.ParquetWriter(output_dir / f"{split}.parquet", schema)
                    writer.write_table(table)
                    count += len(rows)
        finally:
            if writer is not None:
                writer.close()
        print(f"{split}: {count:,} rows -> {output_dir / (split + '.parquet')}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["image", "mixed"], required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    snapshot = snapshot_download(
        repo_id=f"TheToad/IH-{args.variant}", repo_type="dataset",
        allow_patterns=["data/*.parquet"],
    )
    materialize(snapshot, args.output_dir / args.variant)


if __name__ == "__main__":
    main()
