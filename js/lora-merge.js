import {app} from "../../scripts/app.js";

app.registerExtension({
    name: "Comfy.LoRAMerger",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === 'PM LoRA Merger' || nodeData.name === 'PM LoRA SVD Merger') {
            nodeType.prototype.onConnectionsChange = function (type, index, connected) {
                if (type !== 1 ) return;

                this.inputs.forEach((input, i) => input.name = `lora${i + 1}`);

                if (connected && this.inputs[this.inputs.length - 1].link !== null) {
                    this.addInput(`lora${this.inputs.length + 1}`, this.inputs[0].type);
                } else {
                    if (this.inputs.length > 1 && this.inputs[this.inputs.length - 2].link == null)
                        this.removeInput(this.inputs.length - 1);
                }
            }
        }
    },
});