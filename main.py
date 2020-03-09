import hydra

@hydra.main(config_path="./defaults.yaml")
def main(cfg):
    print(cfg)

if __name__=="__main__":
    main()
