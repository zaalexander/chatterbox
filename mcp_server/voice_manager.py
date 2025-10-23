"""
Voice sample management with SQLite storage.

Stores voice sample metadata and manages file storage.
Lightweight: ~1MB database for hundreds of voices.
"""

import sqlite3
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


@dataclass
class VoiceSample:
    """Voice sample metadata."""
    voice_id: str
    name: str
    file_path: str
    language: Optional[str] = None
    description: Optional[str] = None
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    file_size_bytes: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class VoiceManager:
    """
    Manages voice samples with SQLite storage.

    Features:
    - Lightweight SQLite database (~1MB)
    - File storage in local directory
    - Voice metadata (name, language, duration, etc.)
    - Upload validation (size, format)
    """

    def __init__(self, db_path: str = "voices.db", storage_dir: str = "./voices"):
        """
        Args:
            db_path: Path to SQLite database
            storage_dir: Directory to store voice files
        """
        self.db_path = Path(db_path)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._init_db()
        logger.info(f"VoiceManager initialized: {db_path}, {storage_dir}")

    def _init_db(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS voices (
                voice_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                language TEXT,
                description TEXT,
                duration_seconds REAL,
                sample_rate INTEGER,
                file_size_bytes INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_name ON voices(name)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_language ON voices(language)
        """)
        conn.commit()
        conn.close()
        logger.info("Database schema initialized")

    def _generate_voice_id(self, file_path: str) -> str:
        """Generate unique voice ID from file hash."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]

    def add_voice(
        self,
        name: str,
        file_path: str,
        language: Optional[str] = None,
        description: Optional[str] = None,
        max_size_mb: int = 10,
    ) -> VoiceSample:
        """
        Add a new voice sample.

        Args:
            name: Human-readable name
            file_path: Path to audio file
            language: Language code (e.g., 'en', 'fr')
            description: Optional description
            max_size_mb: Maximum file size in MB

        Returns:
            VoiceSample with metadata

        Raises:
            ValueError: If validation fails
        """
        file_path = Path(file_path)

        # Validate file exists
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        # Validate file size
        file_size = file_path.stat().st_size
        max_bytes = max_size_mb * 1024 * 1024
        if file_size > max_bytes:
            raise ValueError(
                f"File too large: {file_size / 1024 / 1024:.1f}MB "
                f"(max: {max_size_mb}MB)"
            )

        # Validate format (basic check)
        if file_path.suffix.lower() not in ['.wav', '.mp3', '.flac', '.ogg']:
            raise ValueError(f"Unsupported format: {file_path.suffix}")

        # Generate voice ID
        voice_id = self._generate_voice_id(str(file_path))

        # Copy file to storage directory
        dest_path = self.storage_dir / f"{voice_id}{file_path.suffix}"
        shutil.copy2(file_path, dest_path)

        # Get audio metadata
        try:
            import torchaudio
            info = torchaudio.info(str(dest_path))
            sample_rate = info.sample_rate
            duration = info.num_frames / sample_rate
        except:
            sample_rate = None
            duration = None
            logger.warning("Could not extract audio metadata")

        # Create voice sample
        now = datetime.utcnow().isoformat()
        voice = VoiceSample(
            voice_id=voice_id,
            name=name,
            file_path=str(dest_path),
            language=language,
            description=description,
            duration_seconds=duration,
            sample_rate=sample_rate,
            file_size_bytes=file_size,
            created_at=now,
            updated_at=now,
        )

        # Save to database
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO voices
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            voice.voice_id,
            voice.name,
            voice.file_path,
            voice.language,
            voice.description,
            voice.duration_seconds,
            voice.sample_rate,
            voice.file_size_bytes,
            voice.created_at,
            voice.updated_at,
        ))
        conn.commit()
        conn.close()

        logger.info(f"Added voice: {name} ({voice_id})")
        return voice

    def get_voice(self, voice_id: str) -> Optional[VoiceSample]:
        """Get voice by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM voices WHERE voice_id = ?",
            (voice_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return VoiceSample(**dict(row))
        return None

    def get_voice_by_name(self, name: str) -> Optional[VoiceSample]:
        """Get voice by name (exact match)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM voices WHERE name = ?",
            (name,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return VoiceSample(**dict(row))
        return None

    def list_voices(
        self,
        language: Optional[str] = None,
        limit: int = 100
    ) -> List[VoiceSample]:
        """
        List all voices.

        Args:
            language: Filter by language
            limit: Maximum results

        Returns:
            List of VoiceSample
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        if language:
            cursor = conn.execute(
                "SELECT * FROM voices WHERE language = ? ORDER BY name LIMIT ?",
                (language, limit)
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM voices ORDER BY name LIMIT ?",
                (limit,)
            )

        voices = [VoiceSample(**dict(row)) for row in cursor.fetchall()]
        conn.close()

        return voices

    def delete_voice(self, voice_id: str) -> bool:
        """
        Delete voice sample.

        Args:
            voice_id: Voice ID to delete

        Returns:
            True if deleted, False if not found
        """
        voice = self.get_voice(voice_id)
        if not voice:
            return False

        # Delete file
        file_path = Path(voice.file_path)
        if file_path.exists():
            file_path.unlink()

        # Delete from database
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM voices WHERE voice_id = ?", (voice_id,))
        conn.commit()
        conn.close()

        logger.info(f"Deleted voice: {voice_id}")
        return True

    def update_voice(
        self,
        voice_id: str,
        name: Optional[str] = None,
        language: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[VoiceSample]:
        """
        Update voice metadata.

        Args:
            voice_id: Voice ID to update
            name: New name (optional)
            language: New language (optional)
            description: New description (optional)

        Returns:
            Updated VoiceSample or None if not found
        """
        voice = self.get_voice(voice_id)
        if not voice:
            return None

        # Update fields
        if name:
            voice.name = name
        if language:
            voice.language = language
        if description:
            voice.description = description

        voice.updated_at = datetime.utcnow().isoformat()

        # Save to database
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE voices
            SET name = ?, language = ?, description = ?, updated_at = ?
            WHERE voice_id = ?
        """, (voice.name, voice.language, voice.description, voice.updated_at, voice_id))
        conn.commit()
        conn.close()

        logger.info(f"Updated voice: {voice_id}")
        return voice

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)

        # Total voices
        total = conn.execute("SELECT COUNT(*) FROM voices").fetchone()[0]

        # Total size
        total_size = conn.execute(
            "SELECT SUM(file_size_bytes) FROM voices"
        ).fetchone()[0] or 0

        # By language
        by_language = {}
        cursor = conn.execute("""
            SELECT language, COUNT(*) as count
            FROM voices
            GROUP BY language
        """)
        for row in cursor:
            lang = row[0] or "unknown"
            by_language[lang] = row[1]

        conn.close()

        return {
            "total_voices": total,
            "total_size_mb": total_size / 1024 / 1024,
            "by_language": by_language,
            "storage_dir": str(self.storage_dir),
            "db_path": str(self.db_path),
        }
