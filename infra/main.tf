data "aws_caller_identity" "current" {}

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  name = "${var.project_name}-${var.environment}"

  # Duas AZs: o ALB exige no mínimo duas, e uma só não sobrevive à queda de zona.
  azs = slice(data.aws_availability_zones.available.names, 0, 2)

  # Nome de bucket é global na AWS, então o account id evita colisão.
  artifacts_bucket = "${local.name}-artifacts-${data.aws_caller_identity.current.account_id}"
  rewards_bucket   = "${local.name}-rewards-${data.aws_caller_identity.current.account_id}"

  container_name = "api"
  container_port = 8000
}
