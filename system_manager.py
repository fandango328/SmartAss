from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import asyncio
from pathlib import Path
import json

class SystemState(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    PROCESSING = "processing"
    ERROR = "error"

@dataclass
class SystemConfig:
    """Stores current system configuration"""
    voice_config: Dict[str, Any]  # Voice settings (ElevenLabs/Local)
    llm_config: Dict[str, Any]    # LLM settings (Anthropic/OpenAI)
    transcription_config: Dict[str, Any]  # Transcription settings
    audio_config: Dict[str, Any]  # Audio processing settings
    debug_mode: bool = False

@dataclass
class CommandType(Enum):
    DOCUMENT = "document"      # File loading/offloading
    TOOL = "tool"             # Tool enabling/disabling
    CALIBRATION = "calibration"  # Voice calibration
    CONFIG = "config"         # System configuration
    CHARACTER = "character"    # Character switching

@dataclass
class SystemConfig:
    """Stores current system configuration"""
    voice_config: Dict[str, Any]  # Voice settings (ElevenLabs/Local)
    llm_config: Dict[str, Any]    # LLM settings (Anthropic/OpenAI)
    transcription_config: Dict[str, Any]  # Transcription settings
    audio_config: Dict[str, Any]  # Audio processing settings
    debug_mode: bool = False

@dataclass
class CommandResult:
    """Stores result of command execution"""
    success: bool
    message: str
    needs_reload: bool = False
    affected_components: List[str] = None
    error: Optional[str] = None
    
class SystemManager:
    def __init__(self, 
                 display_manager, 
                 audio_manager, 
                 document_manager, 
                 token_tracker,
                 config_path: str = "config/system_config.json"):
        self.display_manager = display_manager
        self.audio_manager = audio_manager
        self.document_manager = document_manager
        self.token_tracker = token_tracker
        self.config_path = Path(config_path)
        
        # System state
        self.current_state = SystemState.IDLE
        self.config = self._load_system_config()
        self.command_history = []
        self.error_log = []
        
        # Component status tracking
        self.component_status = {
            "voice": True,
            "llm": True,
            "transcription": True,
            "document": True,
            "tools": True
        }
    
    def _load_system_config(self) -> SystemConfig:
        """Load system configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path) as f:
                    config_data = json.load(f)
                return SystemConfig(**config_data)
            return self._create_default_config()
        except Exception as e:
            self.log_error(f"Error loading config: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> SystemConfig:
        """Create default system configuration"""
        return SystemConfig(
            voice_config={
                "provider": "elevenlabs",
                "voice_id": "L.A.U.R.A.",
                "model": "eleven_flash_v2_5"
            },
            llm_config={
                "provider": "anthropic",
                "model": "claude-3-haiku",
                "temperature": 0.7
            },
            transcription_config={
                "mode": "local",
                "engine": "vosk",
                "vad_settings": {}
            },
            audio_config={
                "sample_rate": 16000,
                "channels": 1,
                "chunk_size": 1024
            }
        )
    
    async def handle_system_command(self, transcript: str) -> CommandResult:
        """Main command handling entry point"""
        try:
            self.current_state = SystemState.PROCESSING
            command_type, action = self._detect_command(transcript.lower())
            
            if not command_type:
                return CommandResult(False, "Not a system command")

            # Execute command based on type
            if command_type == CommandType.DOCUMENT:
                return await self._handle_document_command(action)
            elif command_type == CommandType.TOOL:
                return await self._handle_tool_command(action)
            elif command_type == CommandType.CALIBRATION:
                return await self._handle_calibration_command()
            elif command_type == CommandType.CHARACTER:
                character_name = self._extract_character_name(transcript)
                return await self._handle_character_switch(character_name)
                
        except Exception as e:
            self.log_error(f"Command handling error: {e}")
            return CommandResult(False, "Error executing command", error=str(e))
        finally:
            self.current_state = SystemState.IDLE

    async def _handle_document_command(self, action: str) -> CommandResult:
        """Handle document loading/offloading"""
        try:
            await self.display_manager.update_display('tools')
            
            if action == "load":
                await self.document_manager.load_all_files(clear_existing=False)
                audio_context = ('file', 'loaded')
            else:  # offload
                await self.document_manager.offload_all_files()
                audio_context = ('file', 'offloaded')
                
            # Play audio feedback
            audio_file = self.audio_manager.get_random_audio(*audio_context)
            if audio_file:
                await self.audio_manager.play_audio(audio_file)
                
            await self.audio_manager.wait_for_audio_completion()
            await self.display_manager.update_display('listening')
            
            return CommandResult(
                success=True,
                message=f"Files {action}ed successfully"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"File {action} failed",
                error=str(e)
            )

    async def _handle_tool_command(self, action: str) -> CommandResult:
        """Handle tool enabling/disabling"""
        try:
            await self.display_manager.update_display('tools')
            
            if action == "enable":
                self.token_tracker.enable_tools()
                audio_context = ('tool', 'status/enabled')
            else:  # disable
                self.token_tracker.disable_tools()
                audio_context = ('tool', 'status/disabled')
                
            # Play audio feedback
            audio_file = self.audio_manager.get_random_audio(*audio_context)
            if audio_file:
                await self.audio_manager.play_audio(audio_file)
                
            await self.audio_manager.wait_for_audio_completion()
            await self.display_manager.update_display('listening')
            
            return CommandResult(
                success=True,
                message=f"Tools {action}d successfully"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Tool {action} failed",
                error=str(e)
            )

    async def _handle_calibration_command(self) -> CommandResult:
        """Handle voice calibration"""
        try:
            await self.display_manager.update_display('tools')
            
            calibration_success = await self.audio_manager.run_vad_calibration()
            
            audio_context = ('calibration', 'complete' if calibration_success else 'failed')
            audio_file = self.audio_manager.get_random_audio(*audio_context)
            
            if audio_file:
                await self.audio_manager.play_audio(audio_file)
                
            await self.audio_manager.wait_for_audio_completion()
            await self.display_manager.update_display('listening')
            
            return CommandResult(
                success=calibration_success,
                message="Voice calibration complete" if calibration_success else "Calibration failed"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message="Calibration failed",
                error=str(e)
            )

    async def _handle_character_switch(self, character_name: str) -> CommandResult:
        """Handle character switching"""
        try:
            if character_name not in CHARACTER_CONFIGS:
                return CommandResult(
                    success=False,
                    message=f"Unknown character: {character_name}"
                )
            
            # Get character configuration
            char_config = CHARACTER_CONFIGS[character_name]
            
            # Update voice and system prompt
            updates = {
                "voice": char_config["voice_id"],
                "system_prompt": char_config["system_prompt"]
            }
            
            # Apply updates
            result = await self.update_runtime_config(updates)
            
            if result.success:
                # Play character switch audio
                audio_file = self.audio_manager.get_random_audio("character", character_name.lower())
                if audio_file:
                    await self.audio_manager.play_audio(audio_file)
                    
            return result
            
        except Exception as e:
            return CommandResult(
                success=False,
                message="Character switch failed",
                error=str(e)
            )
    
    async def _reload_components(self, components: List[str]) -> None:
        """Reload specified system components"""
        for component in components:
            try:
                if component == "voice":
                    await self.audio_manager.reload_tts_engine(
                        self.config.voice_config
                    )
                elif component == "llm":
                    self.token_tracker.reload_llm_client(
                        self.config.llm_config
                    )
                elif component == "transcription":
                    await self.audio_manager.reload_transcriber(
                        self.config.transcription_config
                    )
            except Exception as e:
                self.log_error(f"Error reloading {component}: {e}")
    
    def save_system_config(self) -> bool:
        """Save current configuration to file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config.__dict__, f, indent=2)
            return True
        except Exception as e:
            self.log_error(f"Error saving config: {e}")
            return False
    
    def log_error(self, error_msg: str) -> None:
        """Log system errors"""
        self.error_log.append({
            "timestamp": datetime.now().isoformat(),
            "error": error_msg
        })
        if self.config.debug_mode:
            print(f"SystemManager Error: {error_msg}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "state": self.current_state.value,
            "components": self.component_status,
            "config": self.config.__dict__,
            "error_count": len(self.error_log),
            "command_count": len(self.command_history)
        }
        
        

        
async def update_runtime_config(self, updates: Dict[str, Any]) -> CommandResult:
    """
    Update system configuration during runtime
    
    Args:
        updates: Dictionary of configuration updates
        Example: {
            "voice": "MAXIMILIAN",
            "personality": "english_gentleman",
            "system_prompt": "MAXIMILIAN_PROMPT"
        }
    """
    try:
        # Store previous config for rollback
        previous_config = self.config
        affected_components = []
        
        # Update voice configuration
        if "voice" in updates:
            self.config.voice_config["voice_id"] = updates["voice"]
            if "personality" in updates:
                self.config.voice_config["personality"] = updates["personality"]
            affected_components.append("voice")
            
        # Update system prompt
        if "system_prompt" in updates:
            if updates["system_prompt"] in SYSTEM_PROMPTS:
                self.config.voice_config["system_prompt"] = SYSTEM_PROMPTS[updates["system_prompt"]]
                affected_components.append("llm")
                
        # Update LLM model
        if "llm_model" in updates:
            self.config.llm_config["model"] = updates["llm_model"]
            affected_components.append("llm")
            
        # Update TTS provider
        if "tts_provider" in updates:
            self.config.voice_config["provider"] = updates["tts_provider"]
            affected_components.append("voice")
            
        # Attempt to apply changes
        try:
            # Update TTS engine if voice changed
            if "voice" in affected_components:
                await self.audio_manager.update_tts_config(
                    voice_id=self.config.voice_config["voice_id"],
                    provider=self.config.voice_config["provider"]
                )
                
            # Update LLM if system prompt or model changed
            if "llm" in affected_components:
                await self.token_tracker.update_llm_config(
                    model=self.config.llm_config["model"],
                    system_prompt=self.config.voice_config["system_prompt"]
                )
                
            # Save updated config
            self.save_system_config()
            
            return CommandResult(
                success=True,
                message=f"Configuration updated successfully. Changed: {', '.join(affected_components)}",
                needs_reload=False,
                affected_components=affected_components
            )
            
        except Exception as e:
            # Rollback changes on failure
            self.config = previous_config
            raise Exception(f"Failed to apply changes: {str(e)}")
            
    except Exception as e:
        self.log_error(f"Configuration update failed: {e}")
        return CommandResult(
            success=False,
            message="Failed to update configuration",
            error=str(e)
        )

async def hot_swap_character(self, character_name: str) -> CommandResult:
    """
    Hot swap to a different character configuration
    
    Args:
        character_name: Name of character configuration to load
    """
    try:
        # Get character configuration
        if character_name not in CHARACTER_CONFIGS:
            return CommandResult(
                success=False,
                message=f"Character configuration '{character_name}' not found"
            )
            
        character_config = CHARACTER_CONFIGS[character_name]
        
        # Prepare configuration updates
        updates = {
            "voice": character_config["voice_id"],
            "personality": character_config["personality"],
            "system_prompt": character_config["system_prompt"]
        }
        
        # Update configuration
        result = await self.update_runtime_config(updates)
        
        if result.success:
            # Play character switch audio if available
            audio_file = self.audio_manager.get_random_audio("character", character_name)
            if audio_file:
                await self.audio_manager.play_audio(audio_file)
                
            return CommandResult(
                success=True,
                message=f"Successfully switched to character: {character_name}",
                affected_components=result.affected_components
            )
            
        return result
        
    except Exception as e:
        self.log_error(f"Character hot swap failed: {e}")
        return CommandResult(
            success=False,
            message=f"Failed to switch character: {str(e)}"
        )
        
                
