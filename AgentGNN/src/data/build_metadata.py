from pathlib import Path

def build_teccl_metadata(base_path):
    provided_output_files = sorted(
        Path(base_path).rglob("*.json")
    )

    metadata_rows = []

    for path in provided_output_files:
        parts = path.parts

        topo_folder = parts[-5]
        chassis_folder = parts[-4]
        collective = parts[-3]
        mode = parts[-2]
        size_file = path.stem

        if topo_folder.startswith("DGX2"):
            topology_name = "DGX2"
        elif topo_folder.startswith("NDv2"):
            topology_name = "NDv2"
        else:
            topology_name = topo_folder

        chassis = int(chassis_folder.split("_")[0])

        metadata_rows.append({
            "source": "teccl",
            "path": str(path),
            "topology_name": topology_name,
            "chassis": chassis,
            "collective": collective,
            "mode": mode,
            "message_size": size_file,
        })

    return metadata_rows


def build_ccl_metadata():
    ccl_metadata = []

    for topo_type in ["custom"]:
        for collective in ["AllGather", "AlltoAll"]:
            ccl_metadata.append({
                "source": "ccl",
                "topology_name": topo_type,
                "chassis": None,
                "collective": collective,
                "mode": "ILP",
                "message_size": "synthetic",
            })

    return ccl_metadata


def build_all_metadata():
    teccl_path = "external/TE-CCL/teccl/examples/experiments/output_provided"

    teccl_meta = build_teccl_metadata(teccl_path)
    ccl_meta = build_ccl_metadata()

    all_meta = teccl_meta + ccl_meta

    print("Total metadata:", len(all_meta))
    print("Sample:")
    for row in all_meta[:5]:
        print(row)

    return all_meta


if __name__ == "__main__":
    build_all_metadata()