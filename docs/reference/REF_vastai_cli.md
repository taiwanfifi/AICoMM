# Vast.ai CLI Reference

GPU rental platform. Uses CLI to search, create, and manage GPU instances.

## API Key

```
e6fe38173e896880908ce8ded7d4044c16990d616c9744a9414ef7894f8aed6d
```

## Install

```bash
pip install --upgrade vastai
vastai set api-key <api_key>
```

## Search GPU Offers

```bash
vastai search offers 'gpu_ram>=40 reliability>0.95 num_gpus=1' -o 'dph+'
# filters: compute_cap, gpu_ram, num_gpus, reliability, dph ($/hr)
# sort: -o 'field+' asc / -o 'field-' desc
```

## Instance Lifecycle

```bash
vastai create instance <OFFER_ID> --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel --disk 60 --ssh --direct
vastai stop instance <ID>       # pause (keeps data, storage fee only)
vastai start instance <ID>      # resume
vastai reboot instance <ID>
vastai destroy instance <ID>    # irreversible, deletes all data
```

## Connect & Monitor

```bash
ssh $(vastai ssh-url <ID>)
vastai show instances
vastai logs <ID>
vastai label instance <ID> "tag"
vastai execute <ID> 'command'   # remote exec without SSH
```

## File Transfer

```bash
vastai scp-url <ID>
vastai copy <SRC> <DST>         # between instances or local
```
