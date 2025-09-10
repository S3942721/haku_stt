#!/usr/bin/env python3
"""
Manual test script for the End-of-Utterance (EOU) detection model.
This script allows you to input text manually and see the EOU model's output.
"""

import sys
import os

# Add the current directory to Python path to import from ws_stt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ws_stt import EndOfUtteranceDetector

def test_eou_model():
    """Interactive test function for the EOU model"""
    print("=== End-of-Utterance Model Tester ===")
    print("This script tests the TurnSense EOU detection model.")
    print("Enter text to analyze, or 'quit' to exit.")
    print("The model will output the probability of end-of-utterance.\n")
    
    # Initialize the EOU detector (not in quiet mode for testing)
    print("Loading TurnSense EOU model...")
    eou_detector = EndOfUtteranceDetector(quiet_mode=False)
    
    if not eou_detector.tokenizer or not eou_detector.session:
        print("ERROR: Failed to load EOU model. Please check your installation.")
        return
    
    print("Model loaded successfully!\n")
    print("Configuration:")
    print(f"  Threshold: {eou_detector.threshold}")
    print(f"  Minimum words for EOU: {eou_detector.min_words_for_eou}")
    print(f"  Confirmation needed: {eou_detector.confirmation_needed}")
    print()
    
    # Interactive loop
    while True:
        try:
            # Get user input
            text_input = input("Enter text to analyze (or 'quit' to exit): ").strip()
            
            if text_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text_input:
                print("Please enter some text.\n")
                continue
            
            # Analyze the text
            print(f"\nAnalyzing: '{text_input}'")
            print("-" * 50)
            
            # Get word count
            word_count = len(text_input.split())
            print(f"Word count: {word_count}")
            
            # Check minimum word requirement
            if word_count < eou_detector.min_words_for_eou:
                print(f"WARNING: Text has only {word_count} words, but model requires at least {eou_detector.min_words_for_eou} words for EOU detection.")
                print("EOU detection will return False due to insufficient length.\n")
                continue
            
            # Test the EOU detection
            is_eou = eou_detector.detect_eou(text_input)
            
            # Get detailed analysis by calling the internal methods
            try:
                # Format text as the model expects
                formatted_text = f"<|user|> {text_input.strip()} <|im_end|>"
                print(f"Formatted input: '{formatted_text}'")
                
                # Tokenize
                inputs = eou_detector.tokenizer(
                    formatted_text,
                    padding="max_length",
                    max_length=256,
                    return_tensors="pt",
                    truncation=True
                )
                
                # Run inference
                ort_inputs = {
                    'input_ids': inputs['input_ids'].numpy(),
                    'attention_mask': inputs['attention_mask'].numpy()
                }
                
                # Get raw probabilities
                probabilities = eou_detector.session.run(None, ort_inputs)[0]
                eou_probability = float(probabilities[0][1]) if len(probabilities[0]) > 1 else float(probabilities[0][0])
                
                print(f"Raw EOU probability: {eou_probability:.4f}")
                print(f"Threshold: {eou_detector.threshold}")
                print(f"Above threshold: {eou_probability > eou_detector.threshold}")
                
                # Show recent detections history
                print(f"Recent detections history: {eou_detector.recent_detections}")
                print(f"Confirmations in history: {sum(eou_detector.recent_detections[-eou_detector.confirmation_needed:])}")
                
                # Check for natural endings
                text_lower = text_input.strip().lower()
                natural_endings = ['.', '?', '!', '. thank you', '. thanks', 'that\'s it', 'that is it']
                has_natural_ending = any(text_lower.endswith(ending) for ending in natural_endings)
                print(f"Has natural ending: {has_natural_ending}")
                
                print(f"\n>>> FINAL EOU DECISION: {is_eou} <<<")
                
                if is_eou:
                    print("✅ This text is detected as END-OF-UTTERANCE")
                else:
                    print("❌ This text is NOT detected as end-of-utterance")
                
            except Exception as e:
                print(f"Error during detailed analysis: {e}")
                print(f"Basic EOU result: {is_eou}")
            
            print("-" * 50)
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.\n")

def test_batch_examples():
    """Test a batch of predefined examples with full prediction output"""
    print("=== Batch Testing Mode ===")
    print("Testing predefined examples with detailed analysis...\n")
    
    # Initialize the EOU detector
    eou_detector = EndOfUtteranceDetector(quiet_mode=False)
    
    if not eou_detector.tokenizer or not eou_detector.session:
        print("ERROR: Failed to load EOU model.")
        return
    
    # Test examples
    test_cases = [
        # Should NOT be EOU (incomplete thoughts)
        "I think we should",
        "The weather today is",
        "Can you please help me with",
        "I was wondering if",
        "Let me tell you about",
        
        # Should be EOU (complete thoughts)
        "I think we should go to the store today.",
        "The weather today is really nice and sunny.",
        "Can you please help me with this problem? I'm stuck.",
        "I was wondering if you could send me that report. Thank you.",
        "Let me tell you about what happened yesterday. It was amazing!",
        "That's exactly what I needed. Thanks!",
        "Perfect, that sounds good to me.",
        "I'm done with this task now.",
        "Okay, I think that's it for today.",
    ]
    
    print(f"Testing {len(test_cases)} examples:")
    print("=" * 80)
    
    eou_count = 0
    not_eou_count = 0
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{i:2d}. TESTING: '{text}'")
        print("-" * 60)
        
        # Reset the detector's history for each test
        eou_detector.recent_detections = []
        
        word_count = len(text.split())
        print(f"Word count: {word_count}")
        
        # Check minimum word requirement
        if word_count < eou_detector.min_words_for_eou:
            print(f"⚠️  Below minimum word count ({eou_detector.min_words_for_eou}) - EOU will be False")
            print(f"Result: ❌ Not EOU (insufficient words)")
            not_eou_count += 1
            continue
        
        # Get detailed analysis
        try:
            # Format text as the model expects
            formatted_text = f"<|user|> {text.strip()} <|im_end|>"
            
            # Tokenize
            inputs = eou_detector.tokenizer(
                formatted_text,
                padding="max_length",
                max_length=256,
                return_tensors="pt",
                truncation=True
            )
            
            # Run inference
            ort_inputs = {
                'input_ids': inputs['input_ids'].numpy(),
                'attention_mask': inputs['attention_mask'].numpy()
            }
            
            # Get raw probabilities
            probabilities = eou_detector.session.run(None, ort_inputs)[0]
            eou_probability = float(probabilities[0][1]) if len(probabilities[0]) > 1 else float(probabilities[0][0])
            
            print(f"Raw EOU probability: {eou_probability:.4f}")
            print(f"Threshold: {eou_detector.threshold}")
            print(f"Above threshold: {eou_probability > eou_detector.threshold}")
            
            # Check for natural endings
            text_lower = text.strip().lower()
            natural_endings = ['.', '?', '!', '. thank you', '. thanks', 'that\'s it', 'that is it']
            has_natural_ending = any(text_lower.endswith(ending) for ending in natural_endings)
            print(f"Has natural ending: {has_natural_ending}")
            
            # Run the actual EOU detection
            is_eou = eou_detector.detect_eou(text)
            
            print(f"Recent detections: {eou_detector.recent_detections}")
            print(f"Confirmations needed: {eou_detector.confirmation_needed}")
            print(f"Confirmations in history: {sum(eou_detector.recent_detections[-eou_detector.confirmation_needed:])}")
            
            if is_eou:
                print(f">>> ✅ END-OF-UTTERANCE DETECTED <<<")
                eou_count += 1
            else:
                print(f">>> ❌ NOT END-OF-UTTERANCE <<<")
                not_eou_count += 1
                
        except Exception as e:
            print(f"Error during analysis: {e}")
            # Fallback to basic detection
            is_eou = eou_detector.detect_eou(text)
            if is_eou:
                print(f">>> ✅ END-OF-UTTERANCE DETECTED (basic) <<<")
                eou_count += 1
            else:
                print(f">>> ❌ NOT END-OF-UTTERANCE (basic) <<<")
                not_eou_count += 1
    
    print("\n" + "=" * 80)
    print("BATCH TESTING SUMMARY:")
    print(f"Total examples tested: {len(test_cases)}")
    print(f"Detected as EOU: {eou_count}")
    print(f"Detected as NOT EOU: {not_eou_count}")
    print(f"EOU detection rate: {(eou_count/len(test_cases)*100):.1f}%")
    print("=" * 80)

if __name__ == "__main__":
    print("EOU Model Test Script")
    print("Choose testing mode:")
    print("1. Interactive mode (enter text manually)")
    print("2. Batch mode (test predefined examples)")
    print("3. Both modes")
    
    try:
        choice = input("\nEnter choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            test_eou_model()
        elif choice == "2":
            test_batch_examples()
        elif choice == "3":
            test_batch_examples()
            print("\n" + "=" * 60)
            print("Now switching to interactive mode...\n")
            test_eou_model()
        else:
            print("Invalid choice. Running interactive mode by default.")
            test_eou_model()
            
    except KeyboardInterrupt:
        print("\n\nExiting...")