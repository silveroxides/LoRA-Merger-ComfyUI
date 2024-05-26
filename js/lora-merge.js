import { ComfyApp, app } from "../../scripts/app.js";
import { ComfyDialog, $el } from "../../scripts/ui.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
	name: "Comfy.LoRAMerger",

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === 'LoraMerger' || nodeData.name === 'LoraSVDMerger')  {
			var input_name = "lora";

			const onConnectionsChange = nodeType.prototype.onConnectionsChange;
			nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
				if(!link_info)
					return;

				if(type == 2) {
					// connect output
					if(connected && index == 0){

						if(this.outputs[0].type == '*'){
							if(link_info.type == '*') {
								app.graph._nodes_by_id[link_info.target_id].disconnectInput(link_info.target_slot);
							}
							else {
								// propagate type
								this.outputs[0].type = link_info.type;
								this.outputs[0].label = link_info.type;
								this.outputs[0].name = link_info.type;

								for(let i in this.inputs) {
									let input_i = this.inputs[i];
									if(input_i.name != 'select' && input_i.name != 'sel_mode')
										input_i.type = link_info.type;
								}
							}
						}
					}

					return;
				}
				else {
					// connect input
					if(this.inputs[index].name == 'select' || this.inputs[index].name == 'sel_mode')
						return;

					if(this.inputs[0].type == '*'){
						const node = app.graph.getNodeById(link_info.origin_id);
						let origin_type = node.outputs[link_info.origin_slot].type;

						if(origin_type == '*') {
							this.disconnectInput(link_info.target_slot);
							return;
						}

						for(let i in this.inputs) {
							let input_i = this.inputs[i];
							if(input_i.name != 'select' && input_i.name != 'sel_mode')
								input_i.type = origin_type;
						}

						this.outputs[0].type = origin_type;
						this.outputs[0].label = origin_type;
						this.outputs[0].name = origin_type;
					}
				}

				let select_slot = this.inputs.find(x => x.name == "select");
				let mode_slot = this.inputs.find(x => x.name == "sel_mode");

				let converted_count = 0;
				converted_count += select_slot?1:0;
				converted_count += mode_slot?1:0;

				if (!connected && (this.inputs.length > 1+converted_count)) {
					const stackTrace = new Error().stack;

					if(
						!stackTrace.includes('LGraphNode.prototype.connect') && // for touch device
						!stackTrace.includes('LGraphNode.connect') && // for mouse device
						!stackTrace.includes('loadGraphData') &&
						this.inputs[index].name != 'select') {
						this.removeInput(index);
					}
				}

				let slot_i = 1;
				for (let i = 0; i < this.inputs.length; i++) {
					let input_i = this.inputs[i];
					if(input_i.name != 'select'&& input_i.name != 'sel_mode') {
						input_i.name = `${input_name}${slot_i}`
						slot_i++;
					}
				}

				let last_slot = this.inputs[this.inputs.length - 1];
				if (
					(last_slot.name == 'select' && last_slot.name != 'sel_mode' && this.inputs[this.inputs.length - 2].link != undefined)
					|| (last_slot.name != 'select' && last_slot.name != 'sel_mode' && last_slot.link != undefined)) {
						this.addInput(`${input_name}${slot_i}`, this.outputs[0].type);
				}


			}
		}
	},
});
