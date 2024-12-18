# Copyright 2022 The KServe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import os
import json
import pytest
from kubernetes import client
from kserve import (
    constants,
    KServeClient,
    V1beta1InferenceService,
    V1beta1InferenceServiceSpec,
    V1beta1PredictorSpec,
    V1beta1TorchServeSpec,
    V1beta1ModelSpec,
    V1beta1ModelFormat,
)
from kubernetes.client import V1ResourceRequirements, V1ContainerPort, V1Container, V1EnvVar

from ..common.utils import KSERVE_TEST_NAMESPACE, predict_isvc, predict_grpc

@pytest.mark.predictor
@pytest.mark.asyncio(scope="session")
async def test_torchserve_kserve(rest_v1_client):
    service_name = "mnist"
    predictor = V1beta1PredictorSpec(
        min_replicas=1,
        pytorch=V1beta1TorchServeSpec(
            env=[
                V1EnvVar(name="TEMP", value="/tmp"),
            ],
            storage_uri="gs://kfserving-examples/models/torchserve/image_classifier/v1",
            protocol_version="v1",
            resources=V1ResourceRequirements(
                requests={"cpu": "100m", "memory": "256Mi"},
                limits={"cpu": "1", "memory": "1Gi"},
            ),
			ports=[V1ContainerPort(container_port=8085, protocol="TCP")],

        ),
    )


    isvc = V1beta1InferenceService(
        api_version=constants.KSERVE_V1BETA1,
        kind=constants.KSERVE_KIND,
        metadata=client.V1ObjectMeta(
            name=service_name, namespace=KSERVE_TEST_NAMESPACE
        ),
        spec=V1beta1InferenceServiceSpec(predictor=predictor),
    )

    kserve_client = KServeClient(
        config_file=os.environ.get("KUBECONFIG", "~/.kube/config")
    )
    kserve_client.create(isvc)
    kserve_client.wait_isvc_ready(service_name, namespace=KSERVE_TEST_NAMESPACE)

    res = await predict_isvc(
        rest_v1_client, service_name, "./data/torchserve_input.json"
    )
    assert res["predictions"][0] == 2
    kserve_client.delete(service_name, KSERVE_TEST_NAMESPACE)
