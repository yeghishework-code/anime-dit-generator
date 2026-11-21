import argparse
import os
import tarfile
from pathlib import Path, PurePosixPath, PureWindowsPath


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = BASE_DIR / "anime_faces_dataset.tar"
DEFAULT_DATASET_DIR = BASE_DIR / "datasets" / "anime-faces-180k"


def unpack_dataset(archive_path, dataset_dir):
    archive_path = Path(archive_path)
    dataset_dir = Path(dataset_dir)

    if not archive_path.is_file():
        raise FileNotFoundError(f"Архив не найден: {archive_path}")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    destination = dataset_dir.resolve()

    with tarfile.open(archive_path, "r:*") as archive:
        members = archive.getmembers()

        if not members:
            raise RuntimeError(f"Архив пуст: {archive_path}")

        for member in members:
            member_name = member.name.replace("\\", "/")
            member_path = PurePosixPath(member_name)

            if (
                member_path.is_absolute()
                or PureWindowsPath(member.name).is_absolute()
                or PureWindowsPath(member.name).drive
                or ".." in member_path.parts
            ):
                raise ValueError(f"Небезопасный путь в архиве: {member.name}")

            if not (member.isfile() or member.isdir()):
                raise ValueError(
                    f"Неподдерживаемый тип записи в архиве: {member.name}"
                )

            target = (destination / Path(*member_path.parts)).resolve()
            if os.path.commonpath((str(destination), str(target))) != str(destination):
                raise ValueError(f"Путь выходит за каталог датасета: {member.name}")

        archive.extractall(path=destination, members=members)

    extracted_files = sum(1 for member in members if member.isfile())
    return extracted_files


def main():
    parser = argparse.ArgumentParser(
        description="Распаковать архив датасета в каталог для обучения."
    )
    parser.add_argument(
        "archive",
        nargs="?",
        default=DEFAULT_ARCHIVE,
        type=Path,
        help=f"Путь к архиву (по умолчанию: {DEFAULT_ARCHIVE})",
    )
    parser.add_argument(
        "--destination",
        default=DEFAULT_DATASET_DIR,
        type=Path,
        help=f"Каталог распаковки (по умолчанию: {DEFAULT_DATASET_DIR})",
    )
    args = parser.parse_args()

    count = unpack_dataset(args.archive, args.destination)
    print(f"Распаковано файлов: {count}")
    print(f"Каталог датасета: {args.destination.resolve()}")


if __name__ == "__main__":
    main()
