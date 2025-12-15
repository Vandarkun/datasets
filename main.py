from modules.ControllerAgent import DialogueController

PROFILE_FILE = "output/sample_profile.json"
OUTPUT_FILE = "output/sample_dialogue.json"

def main():

    controller = DialogueController(
        profile_path=PROFILE_FILE,
        output_path=OUTPUT_FILE
    )
    
    controller.run()

if __name__ == "__main__":
    main()