
target "slime" {
  contexts = {
  }
  context    = "./"
  dockerfile = "Dockerfile.redmod"
  args       = {}
  tags       = ["slime:latest"]
  secret = [
    "id=wandb_key,src=./secrets/wandb_key",
    "id=huggingface_key,src=./secrets/huggingface_key",
    /* "id=eval_gcp_secret_key,src=./secrets/eval_gcp_key.json", */
    /* "id=azure_secret_key,src=./secrets/azure_creds.json" */
  ]
  /* ssh = ["default"] # Add this line */
}
