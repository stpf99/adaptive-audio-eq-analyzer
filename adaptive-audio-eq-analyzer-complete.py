import numpy as np
import scipy.signal as signal
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import os
import gc
import psutil
import math

def analyze_and_equalize_audio(audio_path, output_path=None, target_sr=None, hifi_mode=True):
    """
    Analizuje plik audio, wykrywa częstotliwości, które wymagają normalizacji
    w pasmach dolnym, średnim i górnym, stosuje korekcję EQ oraz resampluje do wyższej rozdzielczości.
    Automatycznie dostosowuje się do dostępnych zasobów systemu.
    
    Parametry:
    audio_path (str): Ścieżka do pliku audio
    output_path (str, optional): Ścieżka do zapisania poprawionego pliku audio
    target_sr (int, optional): Docelowa częstotliwość próbkowania (Hz). Jeśli None, używany będzie 
                              automatycznie wybrany wyższy sampling rate.
    hifi_mode (bool): Czy używać rozszerzonego pasma dla standardów hi-fi/hi-end
    
    Zwraca:
    tuple: (y_eq, sr) - znormalizowany sygnał i częstotliwość próbkowania
    """
    # Sprawdź dostępne zasoby systemu
    available_memory = psutil.virtual_memory().available
    available_memory_gb = available_memory / (1024**3)
    cpu_cores = psutil.cpu_count(logical=False) or 1
    
    print(f"Dostępna pamięć: {available_memory_gb:.2f} GB")
    print(f"Dostępne rdzenie CPU: {cpu_cores}")
    
    # Wczytanie informacji o pliku audio bez zmiany liczby kanałów i częstotliwości próbkowania
    audio_info = sf.info(audio_path)
    original_sr = audio_info.samplerate
    channels = audio_info.channels
    subtype = audio_info.subtype
    duration = audio_info.duration
    total_frames = audio_info.frames
    
    print(f"Wczytano plik audio: {audio_path}")
    print(f"Oryginalne parametry: {channels} kanały, {original_sr} Hz, format: {subtype}")
    print(f"Czas trwania: {duration:.2f} sekund, liczba próbek: {total_frames}")
    
    # Oszacuj rozmiar danych po resampligu
    if target_sr is None:
        if hifi_mode:
            standard_rates = [96000, 192000, 384000]
            target_sr = next((rate for rate in standard_rates if rate > original_sr), 96000)
        else:
            standard_rates = [44100, 48000, 96000, 192000]
            target_sr = next((rate for rate in standard_rates if rate > original_sr), 96000)
    
    # Oblicz szacunkowe zapotrzebowanie na pamięć (w GB)
    resampling_factor = target_sr / original_sr
    estimated_memory_per_channel = (total_frames * resampling_factor * 4) / (1024**3)  # 4 bajty na float32
    total_estimated_memory = estimated_memory_per_channel * channels * 3  # dla oryginalnych, przetworzonych i tymczasowych danych
    
    print(f"Docelowa częstotliwość próbkowania: {target_sr} Hz")
    print(f"Tryb: {'Hi-Fi/Hi-End' if hifi_mode else 'Standardowy'}")
    print(f"Szacowane zapotrzebowanie na pamięć: {total_estimated_memory:.2f} GB")
    
    # Dostosuj parametry procesowania na podstawie dostępnych zasobów
    use_chunking = total_estimated_memory > available_memory_gb * 0.5
    
    if use_chunking:
        # Oblicz optymalny rozmiar fragmentu (chunk_size) bazując na dostępnej pamięci
        memory_per_second = (original_sr * channels * 4) / (1024**3)  # w GB na sekundę
        chunk_size_seconds = max(5, min(60,(available_memory_gb * 0.3) / memory_per_second))
        chunk_size_frames = int(chunk_size_seconds * original_sr)

        
        print(f"Plik zostanie przetworzony w fragmentach po {chunk_size_seconds:.1f} sekund ({chunk_size_frames} próbek)")
    else:
        chunk_size_frames = total_frames
        print("Plik zostanie przetworzony w całości")
    
    # Dostosuj parametr n_fft w zależności od dostępnej pamięci
    if available_memory_gb < 4:
        base_n_fft = 2048  # Mała pamięć
    elif available_memory_gb < 8:
        base_n_fft = 4096  # Średnia pamięć
    else:
        base_n_fft = 8192  # Duża pamięć
    
    # Przygotuj plik wyjściowy
    if output_path:
        # Określ format na podstawie rozszerzenia pliku wyjściowego
        extension = output_path.split('.')[-1].lower()
        
        # Ustal najlepszy format dla danego rozszerzenia w trybie hi-fi
        if extension == 'wav':
            output_format = 'WAV'
            if hifi_mode:
                output_subtype = 'FLOAT' if 'FLOAT' in subtype else 'PCM_24'
            else:
                output_subtype = 'PCM_24' if 'PCM' in subtype else 'FLOAT'
        elif extension == 'flac':
            output_format = 'FLAC'
            output_subtype = 'PCM_24'
        else:
            output_format = None  # soundfile wybierze odpowiedni format
            output_subtype = None
        
        # Upewnij się, że katalog istnieje
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Przetwarzanie w zależności od wybranej metody
    if use_chunking:
        # Inicjalizacja pliku wyjściowego
        if output_path:
            # Utwórz pusty plik
            empty_chunk = np.zeros((0, channels) if channels > 1 else 0)
            sf.write(output_path, empty_chunk, target_sr, format=output_format, subtype=output_subtype)
        
        processed_chunks = []
        
        for chunk_start in range(0, total_frames, chunk_size_frames):
            chunk_end = min(chunk_start + chunk_size_frames, total_frames)
            chunk_frames = chunk_end - chunk_start
            
            print(f"\nPrzetwarzanie fragmentu {chunk_start/original_sr:.1f}s - {chunk_end/original_sr:.1f}s ({chunk_frames} próbek)")
            
            # Wczytaj tylko bieżący fragment
            y_chunk, sr = sf.read(audio_path, start=chunk_start, frames=chunk_frames)
            
            # Sprawdź liczbę kanałów w fragmentcie
            multi_channel = len(y_chunk.shape) > 1 and y_chunk.shape[1] > 1
            
            # Przetwórz fragment
            if multi_channel:
                num_channels = y_chunk.shape[1]
                
                # Dla każdego kanału
                if original_sr != target_sr:
                    new_length = int(len(y_chunk) * target_sr / original_sr)
                    y_eq_chunk = np.zeros((new_length, num_channels))
                else:
                    y_eq_chunk = np.zeros_like(y_chunk)
                
                for channel in range(num_channels):
                    print(f"\nPrzetwarzanie kanału {channel+1}/{num_channels}")
                    y_channel = y_chunk[:, channel]
                    
                    # Wizualizację wykonujemy tylko dla pierwszego fragmentu
                    visualize = (chunk_start == 0)
                    
                    y_eq_channel = process_single_channel(
                        y_channel, original_sr, target_sr, 
                        f"Kanał {channel+1}", hifi_mode, 
                        n_fft=base_n_fft,
                        visualize=visualize
                    )
                    
                    # Przypisz wynik do odpowiedniego kanału
                    if len(y_eq_channel) == y_eq_chunk.shape[0]:
                        y_eq_chunk[:, channel] = y_eq_channel
                    else:
                        # Obsługa potencjalnej różnicy długości
                        min_len = min(len(y_eq_channel), y_eq_chunk.shape[0])
                        y_eq_chunk[:min_len, channel] = y_eq_channel[:min_len]
                    
                    # Wymuszamy czyszczenie pamięci po każdym kanale
                    gc.collect()
            else:
                # Dla pojedynczego kanału
                visualize = (chunk_start == 0)
                y_eq_chunk = process_single_channel(
                    y_chunk, original_sr, target_sr, 
                    "Mono", hifi_mode,
                    n_fft=base_n_fft,
                    visualize=visualize
                )
                gc.collect()
            
            # Zapisz przetworzony fragment
            if output_path:
                # Dodaj fragment do istniejącego pliku
                sf.write(output_path, y_eq_chunk, target_sr, format=output_format, subtype=output_subtype)
                print(f"Fragment zapisany do {output_path}")
            else:
                processed_chunks.append(y_eq_chunk)
            
            # Czyszczenie pamięci po przetworzeniu fragmentu
            del y_chunk
            gc.collect()
        
        # Jeśli nie zapisujemy do pliku, połącz wszystkie fragmenty
        if not output_path and processed_chunks:
            try:
                y_eq = np.concatenate(processed_chunks, axis=0 if len(processed_chunks[0].shape) == 1 else 0)
                return y_eq, target_sr
            except:
                print("Nie można połączyć wszystkich fragmentów - zwracam listę fragmentów")
                return processed_chunks, target_sr
    else:
        # Przetwarzanie całego pliku na raz
        y, sr = sf.read(audio_path)
        
        # Sprawdzenie liczby kanałów
        multi_channel = len(y.shape) > 1 and y.shape[1] > 1
        
        if multi_channel:
            num_channels = y.shape[1]
            
            # Określ liczbę próbek po resamplingu
            if original_sr != target_sr:
                new_length = int(len(y) * target_sr / original_sr)
                y_eq = np.zeros((new_length, num_channels))
            else:
                y_eq = np.zeros_like(y)
            
            for channel in range(num_channels):
                print(f"\nPrzetwarzanie kanału {channel+1}/{num_channels}")
                y_channel = y[:, channel]
                y_eq_channel = process_single_channel(
                    y_channel, original_sr, target_sr, 
                    f"Kanał {channel+1}", hifi_mode,
                    n_fft=base_n_fft
                )
                
                # Przypisz wynik do odpowiedniego kanału
                if len(y_eq_channel) == y_eq.shape[0]:
                    y_eq[:, channel] = y_eq_channel
                else:
                    # Obsługa potencjalnej różnicy długości
                    min_len = min(len(y_eq_channel), y_eq.shape[0])
                    y_eq[:min_len, channel] = y_eq_channel[:min_len]
                
                # Czyszczenie pamięci po przetworzeniu kanału
                gc.collect()
        else:
            # Dla pojedynczego kanału
            y_eq = process_single_channel(
                y, original_sr, target_sr, 
                "Mono", hifi_mode,
                n_fft=base_n_fft
            )
        
        # Zapisz wynik, jeśli podano ścieżkę
        if output_path:
            sf.write(output_path, y_eq, target_sr, format=output_format, subtype=output_subtype)
            print(f"\nZapisano znormalizowany plik audio: {output_path}")
            print(f"Parametry wyjściowe: {channels} kanały, {target_sr} Hz, format: {output_subtype if output_subtype else 'auto'}")
        
        return y_eq, target_sr

def process_single_channel(y, original_sr, target_sr, channel_name="", hifi_mode=True, n_fft=8192, visualize=True):
    """
    Przetwarza pojedynczy kanał audio - resampluje, analizuje i koryguje EQ z wyspami przyrostowymi.
    
    Parametry:
    y (array): Dane audio jednego kanału
    original_sr (int): Oryginalna częstotliwość próbkowania
    target_sr (int): Docelowa częstotliwość próbkowania
    channel_name (str): Nazwa kanału do wyświetlania w logach
    hifi_mode (bool): Czy używać rozszerzonego pasma dla standardów hi-fi/hi-end
    n_fft (int): Rozmiar okna FFT dla analizy
    visualize (bool): Czy generować wykresy
    
    Zwraca:
    array: Przetworzony kanał audio
    """
    print(f"\nAnaliza {channel_name}:")
    
    # Dostosuj hop_length do rozmiaru n_fft
    hop_length = n_fft // 4
    
    # Resampling do wyższej częstotliwości próbkowania (o ile potrzebny)
    if original_sr != target_sr:
        print(f"Resampling z {original_sr} Hz do {target_sr} Hz...")
        
        # Wybierz odpowiednią metodę resampligu w zależności od rozmiaru danych
        if len(y) > 1000000:  # Dla dużych plików używamy szybszej metody
            try:
                # Bardziej efektywna pamięciowo metoda dla dużych plików
                y_resampled = signal.resample_poly(y, target_sr, original_sr)
            except:
                # Jeśli powyższa metoda zawiedzie, użyj metody librosa, ale z mniejszą precyzją
                y_resampled = librosa.resample(y, orig_sr=original_sr, target_sr=target_sr, res_type='kaiser_fast')
        else:
            # Dla mniejszych plików używamy dokładniejszej metody
            y_resampled = librosa.resample(y, orig_sr=original_sr, target_sr=target_sr)
    else:
        y_resampled = y
    
    # Czyszczenie pamięci po resamplingu
    if original_sr != target_sr:
        del y
        gc.collect()
    
    # Definiujemy zakresy częstotliwości - rozszerzone dla hi-fi
    if hifi_mode:
        # Rozszerzone pasmo dla hi-fi/hi-end
        bass_range = (20, 250)       # rozszerzone pasmo basowe
        mid_range = (250, 5000)      # rozszerzone pasmo średnie
        treble_range = (5000, 20000) # rozszerzone pasmo wysokie
        presence_range = (7000, 12000) # pasmo obecności (presence)
        air_range = (16000, 40000)   # pasmo "powietrza" (hi-end)
    else:
        # Standardowe pasma
        bass_range = (60, 200)       # środkowy zakres basów
        mid_range = (500, 2000)      # środkowy zakres średnich częstotliwości
        treble_range = (5000, 10000) # środkowy zakres wysokich częstotliwości
    
    # Obliczanie STFT (Short-Time Fourier Transform)
    print(f"Wykonuję analizę STFT z rozmiarem okna {n_fft}...")
    stft = librosa.stft(y_resampled, n_fft=n_fft, hop_length=hop_length)
    
    # Konwersja do skali decybelowej
    magnitude = np.abs(stft)
    power_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # Obliczenie częstotliwości dla każdego indeksu FFT
    freqs = librosa.fft_frequencies(sr=target_sr, n_fft=n_fft)
    
    # Funkcja do znajdowania indeksów częstotliwości w zakresie
    def get_freq_indices(f_min, f_max):
        return np.where((freqs >= f_min) & (freqs <= f_max))[0]
    
    # Pobierz indeksy dla każdego zakresu
    bass_indices = get_freq_indices(bass_range[0], bass_range[1])
    mid_indices = get_freq_indices(mid_range[0], mid_range[1])
    treble_indices = get_freq_indices(treble_range[0], treble_range[1])
    
    # Dodatkowe indeksy dla trybu hi-fi
    if hifi_mode:
        presence_indices = get_freq_indices(presence_range[0], presence_range[1])
        air_indices = get_freq_indices(air_range[0], air_range[1])
    
    # Oblicz średnią moc dla każdego zakresu
    bass_power = np.mean(power_db[bass_indices, :], axis=0)
    mid_power = np.mean(power_db[mid_indices, :], axis=0)
    treble_power = np.mean(power_db[treble_indices, :], axis=0)
    
    if hifi_mode:
        presence_power = np.mean(power_db[presence_indices, :], axis=0)
        air_power = np.mean(power_db[air_indices, :], axis=0) if len(air_indices) > 0 else np.array([])
    
    # Oblicz odchylenie standardowe dla każdego zakresu
    bass_std = np.std(power_db[bass_indices, :], axis=0)
    mid_std = np.std(power_db[mid_indices, :], axis=0)
    treble_std = np.std(power_db[treble_indices, :], axis=0)
    
    if hifi_mode:
        presence_std = np.std(power_db[presence_indices, :], axis=0)
        air_std = np.std(power_db[air_indices, :], axis=0) if len(air_indices) > 0 else np.array([])
    
    # Ustal progi dla normalizacji
    std_threshold = 6  # dB
    power_threshold = -20  # dB
    
    # Analiza, które zakresy wymagają normalizacji
    bass_needs_eq = np.mean(bass_std) > std_threshold or np.mean(bass_power) < power_threshold
    mid_needs_eq = np.mean(mid_std) > std_threshold or np.mean(mid_power) < power_threshold
    treble_needs_eq = np.mean(treble_std) > std_threshold or np.mean(treble_power) < power_threshold
    
    if hifi_mode:
        presence_needs_eq = np.mean(presence_std) > std_threshold or np.mean(presence_power) < power_threshold
        air_needs_eq = len(air_indices) > 0 and (np.mean(air_std) > std_threshold or np.mean(air_power) < power_threshold)
    
    print("\nWyniki analizy częstotliwości:")
    print(f"Pasmo dolne ({bass_range[0]}-{bass_range[1]} Hz): Średnia moc: {np.mean(bass_power):.2f} dB, Odchylenie: {np.mean(bass_std):.2f} dB")
    print(f"Pasmo średnie ({mid_range[0]}-{mid_range[1]} Hz): Średnia moc: {np.mean(mid_power):.2f} dB, Odchylenie: {np.mean(mid_std):.2f} dB")
    print(f"Pasmo wysokie ({treble_range[0]}-{treble_range[1]} Hz): Średnia moc: {np.mean(treble_power):.2f} dB, Odchylenie: {np.mean(treble_std):.2f} dB")
    
    if hifi_mode:
        print(f"Pasmo obecności ({presence_range[0]}-{presence_range[1]} Hz): Średnia moc: {np.mean(presence_power):.2f} dB, Odchylenie: {np.mean(presence_std):.2f} dB")
        if len(air_indices) > 0:
            print(f"Pasmo powietrza ({air_range[0]}-{air_range[1]} Hz): Średnia moc: {np.mean(air_power):.2f} dB, Odchylenie: {np.mean(air_std):.2f} dB")
        else:
            print(f"Pasmo powietrza ({air_range[0]}-{air_range[1]} Hz): Brak danych (poza zakresem)")
    
    print("\nZalecenia EQ:")
    if bass_needs_eq:
        print(f"Pasmo dolne: Wymaga korekcji. {'Uwydatnij' if np.mean(bass_power) < power_threshold else 'Obniż'} częstotliwości {bass_range[0]}-{bass_range[1]} Hz")
    else:
        print(f"Pasmo dolne: Dobrze zbalansowane")
        
    if mid_needs_eq:
        print(f"Pasmo średnie: Wymaga korekcji. {'Uwydatnij' if np.mean(mid_power) < power_threshold else 'Obniż'} częstotliwości {mid_range[0]}-{mid_range[1]} Hz")
    else:
        print(f"Pasmo średnie: Dobrze zbalansowane")
        
    if treble_needs_eq:
        print(f"Pasmo wysokie: Wymaga korekcji. {'Uwydatnij' if np.mean(treble_power) < power_threshold else 'Obniż'} częstotliwości {treble_range[0]}-{treble_range[1]} Hz")
    else:
        print(f"Pasmo wysokie: Dobrze zbalansowane")
    
    if hifi_mode:
        if presence_needs_eq:
            print(f"Pasmo obecności: Wymaga korekcji. {'Uwydatnij' if np.mean(presence_power) < power_threshold else 'Obniż'} częstotliwości {presence_range[0]}-{presence_range[1]} Hz")
        else:
            print(f"Pasmo obecności: Dobrze zbalansowane")
            
        if len(air_indices) > 0:
            if air_needs_eq:
                print(f"Pasmo powietrza: Wymaga korekcji. {'Uwydatnij' if np.mean(air_power) < power_threshold else 'Obniż'} częstotliwości {air_range[0]}-{air_range[1]} Hz")
            else:
                print(f"Pasmo powietrza: Dobrze zbalansowane")
    
    # Zastosuj equalizację z sinusoidalnym opadaniem dla wysp przyrostowych
    y_eq = y_resampled.copy()
    nyquist = target_sr / 2
    
    # Funkcja tworząca filtr z sinusoidalnym opadaniem na brzegach
    def create_sine_tapered_filter(freqs, freq_range, gain_db, nyquist, width_factor=0.15):
        # Oblicz szerokość tapera (opadania) jako procent szerokości pasma
        full_width = freq_range[1] - freq_range[0]
        taper_width = full_width * width_factor
        
        # Zakres częstotliwości z opadaniem
        freq_min_taper = freq_range[0]
        freq_max_taper = freq_range[1]
        
        # Utworzenie filtra (początkowo płaskiego)
        filter_gain = np.zeros_like(freqs)
        
        # Indeksy dla różnych części filtra
        inner_indices = np.where((freqs >= freq_min_taper + taper_width) & 
                                (freqs <= freq_max_taper - taper_width))[0]
        left_taper_indices = np.where((freqs >= freq_min_taper) & 
                                    (freqs < freq_min_taper + taper_width))[0]
        right_taper_indices = np.where((freqs > freq_max_taper - taper_width) & 
                                      (freqs <= freq_max_taper))[0]
        
        # Płaskie centrum filtra
        filter_gain[inner_indices] = gain_db
        
        # Sinusoidalne opadanie po lewej stronie (rosnące)
        if len(left_taper_indices) > 0:
            x = np.linspace(0, np.pi/2, len(left_taper_indices))
            taper = np.sin(x) ** 2
            filter_gain[left_taper_indices] = gain_db * taper
        
        # Sinusoidalne opadanie po prawej stronie (malejące)
        if len(right_taper_indices) > 0:
            x = np.linspace(0, np.pi/2, len(right_taper_indices))
            taper = np.sin(np.pi/2 - x) ** 2
            filter_gain[right_taper_indices] = gain_db * taper
        
        return filter_gain
    
    # Inicjalizacja filtra EQ
    eq_filter = np.zeros_like(freqs)
    
    # Tworzenie filtrów dla poszczególnych pasm z sinusoidalnym opadaniem
    if bass_needs_eq:
        bass_gain = 6 if np.mean(bass_power) < power_threshold else -6
        bass_filter = create_sine_tapered_filter(freqs, bass_range, bass_gain, nyquist)
        eq_filter += bass_filter
        print(f"Zastosowano EQ z sinusoidalnym opadaniem dla pasma dolnego: {bass_gain} dB")
    
    if mid_needs_eq:
        mid_gain = 3 if np.mean(mid_power) < power_threshold else -3
        mid_filter = create_sine_tapered_filter(freqs, mid_range, mid_gain, nyquist)
        eq_filter += mid_filter
        print(f"Zastosowano EQ z sinusoidalnym opadaniem dla pasma średniego: {mid_gain} dB")
    
    if treble_needs_eq:
        treble_gain = 4 if np.mean(treble_power) < power_threshold else -4
        treble_filter = create_sine_tapered_filter(freqs, treble_range, treble_gain, nyquist)
        eq_filter += treble_filter
        print(f"Zastosowano EQ z sinusoidalnym opadaniem dla pasma wysokiego: {treble_gain} dB")
    
    if hifi_mode:
        if presence_needs_eq:
            presence_gain = 3 if np.mean(presence_power) < power_threshold else -3
            presence_filter = create_sine_tapered_filter(freqs, presence_range, presence_gain, nyquist)
            eq_filter += presence_filter
            print(f"Zastosowano EQ z sinusoidalnym opadaniem dla pasma obecności: {presence_gain} dB")
        
        if len(air_indices) > 0 and air_needs_eq:
            air_gain = 2 if np.mean(air_power) < power_threshold else -2
            air_filter = create_sine_tapered_filter(freqs, air_range, air_gain, nyquist)
            eq_filter += air_filter
            print(f"Zastosowano EQ z sinusoidalnym opadaniem dla pasma powietrza: {air_gain} dB")
    
    # Zastosowanie filtra EQ z wyspami przyrostowymi
    if np.any(eq_filter != 0):
        # Konwersja filtra z dB na współczynniki liniowe
        eq_filter_linear = 10 ** (eq_filter / 20)
        
        # Zastosowanie filtra w dziedzinie częstotliwości
        stft_filtered = stft.copy()
        for i in range(stft.shape[1]):
            stft_filtered[:, i] = stft[:, i] * eq_filter_linear
        
        # Konwersja z powrotem do dziedziny czasu
        y_eq = librosa.istft(stft_filtered, hop_length=hop_length, length=len(y_resampled))
    else:
        print("Nie wykryto potrzeby korekcji EQ")
    
    # Normalizacja głośności wyniku z zachowaniem dynamiki
    peak = np.max(np.abs(y_eq))
    if peak > 0.98:  # Zapobiegamy clippingowi
        y_eq = y_eq * (0.98 / peak)
    
    # Wizualizacja tylko jeśli potrzebna
    if visualize:
        try:
            plt.figure(figsize=(14, 12))
            
            # Oryginalny spektrogram (użyj podpróbkowania dla wykresu)

            plot_step = max(1, len(power_db[0]) // 1000)

            plt.subplot(3, 1, 1)
            librosa.display.specshow(
                power_db[:, ::plot_step],
                sr=target_sr,
                hop_length=hop_length,
                x_axis='time',
                y_axis='log'
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Spektrogram oryginalny {channel_name}')

            # Zastosowany filtr EQ
            plt.subplot(3, 1, 2)
            plt.semilogx(freqs, eq_filter)
            plt.grid(True, which="both", ls="--", alpha=0.7)
            plt.xlabel('Częstotliwość (Hz)')
            plt.ylabel('Wzmocnienie (dB)')
            plt.title('Zastosowany filtr EQ')
            plt.xlim(20, nyquist)

            # Przefiltrowany spektrogram
            plt.subplot(3, 1, 3)
            stft_filtered_db = librosa.amplitude_to_db(np.abs(stft_filtered), ref=np.max)
            librosa.display.specshow(
                stft_filtered_db[:, ::plot_step],
                sr=target_sr,
                hop_length=hop_length,
                x_axis='time',
                y_axis='log'
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Spektrogram po korekcji EQ {channel_name}')

            plt.tight_layout()

            # Gdy używamy fragmentacji, zapisujemy tylko wykres dla pierwszego fragmentu
            plt.savefig(f'eq_analysis_{channel_name.replace(" ", "_")}.png', dpi=150)
            plt.close()

            print(f"Zapisano wizualizację EQ dla {channel_name}")
        except Exception as e:
            print(f"Nie można wygenerować wizualizacji: {e}")

    # Zwróć znormalizowany sygnał
    return y_eq

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analiza i korekcja EQ plików audio.')
    parser.add_argument('input_file', type=str, help='Ścieżka do pliku audio wejściowego.')
    parser.add_argument('--output', '-o', type=str, help='Ścieżka do pliku wyjściowego.', default=None)
    parser.add_argument('--sample-rate', '-sr', type=int, help='Docelowa częstotliwość próbkowania.', default=None)
    parser.add_argument('--standard', '-s', action='store_false', dest='hifi', help='Użyj standardowych pasm zamiast hi-fi.')

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Błąd: Plik {args.input_file} nie istnieje.")
        exit(1)

    # Domyślna nazwa pliku wyjściowego, jeśli nie podano
    if args.output is None:
        base_name = os.path.splitext(args.input_file)[0]
        ext = os.path.splitext(args.input_file)[1]
        args.output = f"{base_name}_eq{ext}"

    # Wywołanie głównej funkcji
    analyze_and_equalize_audio(
        args.input_file,
        output_path=args.output,
        target_sr=args.sample_rate,
        hifi_mode=args.hifi
    )
