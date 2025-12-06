log_file_base="gemma3-$(date +%Y%m%d-%H%M%S)"
stdout_log_file="${log_file_base}-stdout.log"
stderr_log_file="${log_file_base}-stderr.log"

screen -S quantizing -dm bash -c "source .venv/bin/activate; python3 quantizer.py > >(tee \"$stdout_log_file\") 2> >(tee \"$stderr_log_file\" >&2)"
