# PyTorch load IPFS model

See the jupyter notepad for how it works 

You need to be running a local IPFS daemon (`ipfs daemon`) or passing a valid IPFS url

        model = load_state_dict_from_ipfs('a-models-cid-goes-here', model_dir="./my-models", file_name="some-fancy-model.pth")
        print(model)
